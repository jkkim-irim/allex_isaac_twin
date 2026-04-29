"""Realtime per-joint torque plotter for the ALLEX digital twin.

This module is the **client** side of a two-process design:

- Isaac Sim Kit process (this file): samples τ_total each physics step and
  pushes JSON lines into the stdin of a subprocess. **No** matplotlib or
  Tk import happens in the Kit process.
- Subprocess (``plot_proc.py``): runs under a *system* Python
  (``/usr/bin/python3`` or similar) that ships ``_tkinter``, owns the Tk
  mainloop and ``matplotlib.animation.FuncAnimation``.

Rationale:
* Isaac Sim 5.x bundled Python lacks ``_tkinter.so``.
* Qt-based matplotlib backends can deadlock / clash with Kit's own Qt
  event loop, so they are not an option.

IPC is line-delimited JSON. At ~200 Hz with ~20 joints, the payload is
well under 20 KB/s — JSON overhead is negligible.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger("allex.torque_plotter")


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
_DEFAULT_WINDOW_SECONDS = 5.0
_DEFAULT_PHYSICS_HZ = 200.0

# Regex patterns selecting the "body" joint set (arm L/R + waist + neck).
_BODY_RE = re.compile(r"Shoulder|Elbow|Wrist|Waist|Neck", re.IGNORECASE)
_HAND_RE = re.compile(r"Thumb|Index|Middle|Ring|Little", re.IGNORECASE)


# ---------------------------------------------------------------------------
# External Python resolution
# ---------------------------------------------------------------------------
_PROBE_SNIPPET = "import tkinter, matplotlib, numpy"  # noqa: S608 (not SQL)


def _probe_python(py: str, timeout: float = 3.0) -> bool:
    """Return True iff ``py -c 'import tkinter, matplotlib, numpy'`` succeeds."""
    try:
        res = subprocess.run(
            [py, "-c", _PROBE_SNIPPET],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
        return res.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _resolve_plotter_python() -> Optional[Path]:
    """Locate a system Python that has tkinter + matplotlib + numpy importable.

    Priority:
      1. ``$ALLEX_PLOTTER_PYTHON``
      2. ``shutil.which("python3")``
      3. ``/usr/bin/python3``

    Each candidate is probed via a 3 s subprocess import test. First that
    passes wins. Returns ``None`` if none qualify.
    """
    candidates: list[str] = []
    env_override = os.environ.get("ALLEX_PLOTTER_PYTHON")
    if env_override:
        candidates.append(env_override)
    which_py = shutil.which("python3")
    if which_py:
        candidates.append(which_py)
    candidates.append("/usr/bin/python3")

    tried: list[str] = []
    for c in candidates:
        if c in tried:
            continue
        tried.append(c)
        if _probe_python(c):
            return Path(c)
    return None


def _to_np(t) -> Optional[np.ndarray]:
    """Coerce warp/torch/numpy scalars to a flat numpy float32 array.

    Mirrors the pattern from asset_manager.py.
    """
    if t is None:
        return None
    try:
        if hasattr(t, "detach"):
            t = t.detach().cpu().numpy()
        return np.asarray(t, dtype=np.float32).reshape(-1)
    except Exception as exc:
        logger.debug(f"_to_np failed: {exc}")
        return None


def _group_for_joint(name: str) -> str:
    """Return 'arm_L', 'arm_R', or 'other' (waist/neck) for subplot routing."""
    if name.startswith("L_"):
        return "arm_L"
    if name.startswith("R_"):
        return "arm_R"
    return "other"


def _build_dof_name_to_abbr_lut() -> dict[str, str]:
    """dof_name (== USD joint name) → ROS2Config abbr (e.g. "R11", "LSP", "WY", "NP").

    Three sources are merged. Lookup order = source order, but values are
    consistent across sources (sanity-checked at module import indirectly
    via overlap).

    Sources:
      1. ``viz_config.HAND_JOINT_TORQUE_RING_MAP`` — covers 40 hand joints
         (incl. mimic-slave DIP/IP) + 14 arm joints.
      2. ``ALLEX_CSV_JOINT_NAMES`` × ``ROS2Config.OUTBOUND_TOPIC_TO_JOINTS``
         — covers waist (WY, WP) and neck (NP, NY); same topic key gives
         dof_name list and abbr list, zipped index-wise.
    """
    lut: dict[str, str] = {}

    # Source 1: viz_config HAND_JOINT_TORQUE_RING_MAP
    try:
        from ..config.viz_config import HAND_JOINT_TORQUE_RING_MAP
        for entry in HAND_JOINT_TORQUE_RING_MAP:
            j = entry.get("usd_joint_name")
            a = entry.get("dof_abbr")
            if j and a:
                lut[j] = a
    except Exception as exc:
        logger.debug(f"viz_config import failed in LUT build: {exc}")

    # Source 2: joint_name_map × ros2_config — for waist/neck (and overlap
    # confirmation for arm/hand).
    try:
        from .ros2_settings_utils import ROS2Config
        from ..trajectory_generate.joint_name_map import ALLEX_CSV_JOINT_NAMES
        for topic_key, abbr_list in ROS2Config.OUTBOUND_TOPIC_TO_JOINTS.items():
            dof_list = ALLEX_CSV_JOINT_NAMES.get(topic_key)
            if not dof_list:
                continue
            for dof_name, abbr in zip(dof_list, abbr_list):
                # Only set if not already set by source 1; both should agree.
                lut.setdefault(dof_name, abbr)
    except Exception as exc:
        logger.debug(f"ros2/joint_name_map import failed in LUT build: {exc}")

    return lut


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class TorquePlotter:
    """Subprocess-backed realtime torque plotter.

    Parameters
    ----------
    articulation : SingleArticulation
        Must expose ``dof_names`` and ``_articulation_view`` (used by the
        per-step sampler).
    ff_provider : callable returning np.ndarray of shape (num_dof,), optional
        Invoked on the main thread during ``apply_step`` to fetch the
        most-recent applied feedforward torque. Typically
        ``FeedforwardTorqueManager.get_last``.
    subset : "body" | "hand"
    physics_hz : float
        Passed to the subprocess as the assumed sample rate when sizing
        the rolling buffer.
    window_s : float
        Visible time window (seconds).
    """

    def __init__(
        self,
        articulation,
        ff_provider: Optional[Callable[[], np.ndarray]] = None,
        subset: str = "body",
        physics_hz: float = _DEFAULT_PHYSICS_HZ,
        window_s: float = _DEFAULT_WINDOW_SECONDS,
        plot_hz: float = 50.0,
        ff_manager=None,
        plot_mode: str = "rolling",
        save_on_exit: bool = False,
        real_provider: Optional[Callable[[], dict]] = None,
    ):
        self._articulation = articulation
        self._ff_provider = ff_provider
        # FeedforwardTorqueManager — ``apply_step`` 에서 ``read_current()`` 로
        # control.joint_f 의 ground-truth FF 를 읽는 데 사용.
        self._ff_manager = ff_manager
        # ROS2 측 real torque snapshot provider. ``dict[joint_abbr -> float]``
        # 또는 ``None`` 반환. ``None`` 이거나 callable 이 None 이면 plotter 는
        # real 라인을 비활성화한다 (subprocess 에 ``has_real=False`` 통보).
        self._real_provider = real_provider
        self._subset = subset
        self._physics_hz = float(physics_hz)
        self._window_s = float(window_s)
        self._plot_hz = float(plot_hz)
        # "rolling" : 짧은 window + autoscale (현재 동작).
        # "cumulative": 시작부터 모든 데이터 누적, x 축은 0~now 로 계속 확장.
        self._plot_mode = "cumulative" if str(plot_mode).lower() == "cumulative" else "rolling"
        self._save_on_exit = bool(save_on_exit)

        # Resolve DOF names + filter to subset.
        try:
            dof_names = list(articulation.dof_names or [])
        except Exception:
            dof_names = []
        self._dof_names_all = dof_names
        self._num_dof = len(dof_names)
        self._dof_name_to_idx = {n: i for i, n in enumerate(dof_names)}

        self._plot_indices, self._plot_names = self._select_joints(dof_names, subset)
        self._groups = self._build_groups(self._plot_names, subset)
        # Flat joint order matches how tau is serialised to the subprocess.
        self._joint_order: list[str] = [j for g in self._groups for j in g["joints"]]

        # dof_name (== USD joint name) → abbr (ROS2Config) LUT. Used to
        # resolve ``real_provider()`` dict keys for each joint in
        # ``_joint_order``. Built once; missing entries silently fall back to
        # 0.0 in ``apply_step``.
        self._dof_name_to_abbr: dict[str, str] = _build_dof_name_to_abbr_lut()
        # Pre-resolved abbr list aligned with ``_joint_order`` for hot path.
        # ``None`` for joints without a known abbr.
        self._joint_order_abbrs: list[Optional[str]] = [
            self._dof_name_to_abbr.get(name) for name in self._joint_order
        ]
        # Warn once on missing abbrs — useful when LUT sources drift from
        # articulation dof_names.
        missing = [n for n, a in zip(self._joint_order, self._joint_order_abbrs)
                   if a is None]
        if missing:
            logger.warning(
                f"TorquePlotter({subset}): {len(missing)} joint(s) have no abbr "
                f"mapping; real-torque line will be 0 for them: {missing[:5]}..."
            )

        # Subprocess state.
        self._proc: Optional[subprocess.Popen] = None
        self._stdin_lock = threading.Lock()
        self._sim_time: float = 0.0

        # CSV replayer 등 외부 소스에서 sim/real torque 를 직접 공급할 때 쓰는 버퍼.
        # apply_step 가 articulation 폴링 대신 이 값을 쓴다. None 이면 기존 동작 유지.
        # _external_sim_tau     : (num_dof,) numpy float32 — 총 sim torque (PD+FF 합산 결과로 간주).
        # _external_real_by_abbr: dict[abbr -> tau] — real_provider 와 동일 형식.
        self._external_sim_tau: Optional[np.ndarray] = None
        self._external_real_by_abbr: Optional[dict] = None

        # Decimate physics step → plotter sampling. FuncAnimation renders at
        # 20 Hz so much higher than ~50 Hz is wasted GPU→CPU sync.
        self._decim = self._compute_decim(self._plot_hz)
        self._step_count = 0

    def set_plot_mode(self, mode: str) -> None:
        """Pre-start 만 의미 있음 — Start 후 변경은 다음 spawn 부터 반영."""
        self._plot_mode = "cumulative" if str(mode).lower() == "cumulative" else "rolling"

    @property
    def plot_mode(self) -> str:
        return self._plot_mode

    def set_save_on_exit(self, on: bool) -> None:
        """Subprocess 종료 시 자동으로 CSV+PNG 저장할지. Pre-start 만 의미 있음."""
        self._save_on_exit = bool(on)

    @property
    def save_on_exit(self) -> bool:
        return self._save_on_exit

    def _compute_decim(self, plot_hz: float) -> int:
        hz = max(1.0, float(plot_hz))
        return max(1, int(round(self._physics_hz / hz)))

    def set_plot_hz(self, plot_hz: float) -> None:
        """Change sampling rate at runtime. Effective immediately; no restart."""
        self._plot_hz = max(1.0, float(plot_hz))
        self._decim = self._compute_decim(self._plot_hz)

    def set_external_sim_tau(self, arr) -> None:
        """CSV replay 동안 articulation 폴링 대신 사용할 (num_dof,) sim torque array.

        ``arr`` is expected to be a numpy reference; replayer mutates in place every
        step and plotter reads the latest values. ``None`` clears the override.
        """
        if arr is None:
            self._external_sim_tau = None
            return
        self._external_sim_tau = np.asarray(arr, dtype=np.float32).reshape(-1)

    def set_external_real_by_abbr(self, d) -> None:
        """CSV replay 동안 real_provider 대신 사용할 ``dict[abbr -> tau]``.

        Replayer is expected to mutate the same dict in place per step. ``None``
        clears the override.
        """
        self._external_real_by_abbr = None if d is None else d

    @property
    def plot_hz(self) -> float:
        return self._plot_hz

    # ------------------------------------------------------------------
    # Joint / group selection
    # ------------------------------------------------------------------
    @staticmethod
    def _select_joints(dof_names: list[str], subset: str) -> tuple[list[int], list[str]]:
        pat = _HAND_RE if subset == "hand" else _BODY_RE
        indices: list[int] = []
        names: list[str] = []
        for i, n in enumerate(dof_names):
            if pat.search(n):
                indices.append(i)
                names.append(n)
        return indices, names

    @staticmethod
    def _build_groups(plot_names: list[str], subset: str) -> list[dict]:
        """Partition selected joints into display groups."""
        if subset == "hand":
            hand_L = [n for n in plot_names if n.startswith("L_")]
            hand_R = [n for n in plot_names if n.startswith("R_")]
            out: list[dict] = []
            if hand_L:
                out.append({"name": "Hand L", "joints": hand_L})
            if hand_R:
                out.append({"name": "Hand R", "joints": hand_R})
            return out
        buckets: dict[str, list[str]] = {"arm_L": [], "arm_R": [], "other": []}
        for n in plot_names:
            buckets[_group_for_joint(n)].append(n)
        out = []
        if buckets["arm_L"]:
            out.append({"name": "Arm L", "joints": buckets["arm_L"]})
        if buckets["arm_R"]:
            out.append({"name": "Arm R", "joints": buckets["arm_R"]})
        if buckets["other"]:
            out.append({"name": "Waist + Neck", "joints": buckets["other"]})
        return out

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        if self._proc is None:
            return False
        if self._proc.poll() is not None:
            # Subprocess already exited.
            self._proc = None
            return False
        return True

    def start(self) -> bool:
        if self.is_running():
            logger.info("TorquePlotter already running")
            return True
        if not self._plot_indices or not self._groups:
            logger.warning(
                f"TorquePlotter: no joints matched subset={self._subset}; not starting"
            )
            return False

        py = _resolve_plotter_python()
        if py is None:
            msg = (
                "[ALLEX][TorquePlot] no external python found with "
                "(tkinter, matplotlib, numpy). Install: "
                "sudo apt install python3-tk python3-matplotlib python3-numpy, "
                "or set ALLEX_PLOTTER_PYTHON to a suitable interpreter."
            )
            print(msg)
            logger.error(msg)
            return False

        script = Path(__file__).parent / "plot_proc.py"
        if not script.is_file():
            logger.error(f"TorquePlotter: proc script missing: {script}")
            return False

        try:
            self._proc = subprocess.Popen(
                [str(py), str(script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=sys.stderr,
                text=True,
                bufsize=1,  # line-buffered
            )
        except Exception as exc:
            logger.error(f"TorquePlotter: Popen failed: {exc}")
            self._proc = None
            return False

        init_msg = {
            "cmd": "init",
            "title": f"ALLEX Torque ({self._subset}) [{self._plot_mode}]",
            "window_sec": self._window_s,
            "hz": self._physics_hz,
            "y_label": "torque [N m]",
            "plot_mode": self._plot_mode,
            "save_on_exit": self._save_on_exit,
            "groups": self._groups,
            # Real-torque second line (ROS2-fed). When False, subprocess
            # skips dashed real-line creation and ignores any "y_real" payload.
            "has_real": self._real_provider is not None,
        }
        if not self._send(init_msg):
            logger.error("TorquePlotter: init handshake write failed")
            self.stop(timeout=1.0)
            return False

        logger.info(
            f"TorquePlotter started (subset={self._subset}, "
            f"joints={len(self._plot_indices)}, python={py})"
        )
        return True

    def stop(self, timeout: float = 2.0) -> None:
        proc = self._proc
        if proc is None:
            return
        # Ask subprocess to quit cleanly.
        try:
            self._send({"cmd": "quit"})
        except Exception:
            pass
        # Close stdin so EOF triggers exit in subprocess.
        try:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._proc = None
        logger.info("TorquePlotter stopped")

    # ------------------------------------------------------------------
    # Per-step sampling (main / physics thread)
    # ------------------------------------------------------------------
    def apply_step(self, dt: float) -> None:
        """Called on Isaac Sim main thread per physics step.

        Reads the simulation's actual applied torque per DOF and pushes a sample
        frame. Silent no-op when subprocess isn't running — this lets
        ``scenario._push_torque_sample`` call both body/hand plotters
        unconditionally.

        소스: Newton MuJoCo solver 가 실제 joint 에 가하는 총 torque.

          τ_total_sim  =  state_0.mujoco.qfrc_actuator   (PD clipping 후 출력)
                       +  control.joint_f                 (사용자 feedforward)

        PD 재계산 (kp*Δp + kd*Δv + τ_ff) 경로는 수치가 실제 solver 내부 clipping /
        limits / equality constraint 보정 전이라 "simulation 이 진짜 내는 힘" 과
        다를 수 있다. 아래처럼 state 에서 직접 읽는 편이 진실에 가까움.
        """
        self._step_count += 1
        if not self.is_running():
            return
        if not self._plot_indices:
            return
        # Throttle GPU→CPU reads to the plotter's render rate.
        if self._step_count % self._decim != 0:
            return

        num = self._num_dof
        tau_total: Optional[np.ndarray] = None

        # --- 외부 소스 (CSV replay 등) override 우선 ---
        ext_sim = self._external_sim_tau
        if ext_sim is not None and ext_sim.size >= num:
            tau_total = ext_sim[:num].astype(np.float32, copy=True)

        if tau_total is None:
            # --- qfrc_actuator (PD clipping 후 solver 가 쓰는 실제 torque) ----
            # Wrapper: SingleArticulation.get_applied_joint_efforts() 가 패치된
            # articulation_view.get_dof_actuation_forces() 를 통해 내부적으로
            # state_0.mujoco.qfrc_actuator 를 읽어옴.
            tau_pd_act = None
            try:
                arr = self._articulation.get_applied_joint_efforts()
                if arr is not None:
                    a = _to_np(arr)
                    if a is not None and a.ndim == 2:
                        a = a[0]
                    if a is not None:
                        tau_pd_act = np.asarray(a, dtype=np.float32).reshape(-1)
            except Exception as exc:
                if self._step_count % (200 * self._decim) == 0:
                    print(f"[ALLEX][TorquePlot {self._subset}] "
                          f"get_applied_joint_efforts failed: {exc}")

            # --- joint_f (ground-truth FF, 모든 writer 합산 결과) -------------
            # Wrapper: FeedforwardTorqueManager.read_current() — control.joint_f 를
            # 직접 읽어 return. 여러 writer (this manager, 외부 script 등) 의
            # 최종 합산 상태.
            tau_ff = np.zeros(num, dtype=np.float32)
            ff_mgr = self._ff_manager
            if ff_mgr is not None:
                try:
                    arr = ff_mgr.read_current()
                    if arr is not None and arr.size >= num:
                        tau_ff = arr[:num]
                except Exception as exc:
                    if self._step_count % (200 * self._decim) == 0:
                        print(f"[ALLEX][TorquePlot {self._subset}] "
                              f"ff_manager.read_current failed: {exc}")

            # qfrc_actuator 가 아직 채워지지 않은 초기 몇 step 은 skip.
            if tau_pd_act is None or tau_pd_act.size < num:
                if self._step_count % (200 * self._decim) == 0:
                    size = None if tau_pd_act is None else int(tau_pd_act.size)
                    print(f"[ALLEX][TorquePlot {self._subset}] qfrc_actuator unavailable "
                          f"(size={size}, need>={num}). "
                          f"Newton patch 가 빠졌거나 아직 build 전일 수 있음.")
                return

            tau_total = (tau_pd_act[:num] + tau_ff[:num]).astype(np.float32)

        # --- (legacy, 참고용) PD 재계산 경로 — 주석 보존 ----------------
        # try:
        #     view = getattr(self._articulation, "_articulation_view", None)
        #     pos = _to_np(view.get_joint_positions())
        #     vel = _to_np(view.get_joint_velocities())
        #     kps, kds = view.get_gains()
        #     kp_vec = _to_np(kps)
        #     kd_vec = _to_np(kds)
        #     actions = view.get_applied_actions(clone=False)
        #     pos_tgt = _to_np(actions.joint_positions)
        #     vel_tgt = _to_np(actions.joint_velocities)
        # except Exception:
        #     pass
        # tau_pd  = kp_vec * (pos_tgt - pos) + kd_vec * (vel_tgt - vel)
        # tau_ff_cb = self._ff_provider() if self._ff_provider else 0
        # tau_total = (tau_pd + tau_ff_cb).astype(np.float32)

        effective_dt = (float(dt) if dt else (1.0 / self._physics_hz)) * self._decim
        self._sim_time += effective_dt

        # --- real torque snapshot (ROS2-fed 또는 외부 override) ----------------
        # 우선순위: external_real_by_abbr (CSV replay) → real_provider → None.
        tau_real_list: Optional[list[float]] = None
        ext_real = self._external_real_by_abbr
        if ext_real is not None:
            tau_real_list = [
                float(ext_real.get(abbr, 0.0)) if abbr is not None else 0.0
                for abbr in self._joint_order_abbrs
            ]
        elif self._real_provider is not None:
            try:
                snap = self._real_provider()
            except Exception as exc:
                if self._step_count % (200 * self._decim) == 0:
                    print(f"[ALLEX][TorquePlot {self._subset}] "
                          f"real_provider failed: {exc}")
                snap = None
            if isinstance(snap, dict):
                tau_real_list = [
                    float(snap.get(abbr, 0.0)) if abbr is not None else 0.0
                    for abbr in self._joint_order_abbrs
                ]
            else:
                tau_real_list = [0.0] * len(self._joint_order)

        self.push_sample(self._sim_time, tau_total, self._dof_name_to_idx,
                         tau_real_list=tau_real_list)

    def push_sample(
        self,
        t: float,
        tau_full: np.ndarray,
        dof_name_to_idx: dict,
        tau_real_list: Optional[list] = None,
    ) -> None:
        """Send one sample frame to the subprocess.

        ``tau_full`` is expected to be shape (num_dof,). The values are
        picked in ``self._joint_order`` order (which matches the init
        handshake). Missing joints are skipped; the subprocess enforces
        length equality and drops mismatched frames.

        ``tau_real_list`` (optional) is a length-``len(_joint_order)`` list
        of ROS2-fed real torque values, already pre-aligned. Sent under
        ``y_real`` only when not None and ``has_real`` was set in init.
        """
        if not self.is_running():
            return
        tau_list: list[float] = []
        try:
            for name in self._joint_order:
                idx = dof_name_to_idx.get(name)
                if idx is None or idx >= tau_full.size:
                    # Cannot safely align; skip this frame entirely rather
                    # than send a mismatched-length payload.
                    return
                tau_list.append(float(tau_full[idx]))
        except Exception as exc:
            logger.debug(f"push_sample serialise failed: {exc}")
            return
        frame: dict = {"t": float(t), "y": tau_list}
        if tau_real_list is not None and len(tau_real_list) == len(tau_list):
            frame["y_real"] = [float(v) for v in tau_real_list]
        self._send(frame)

    # ------------------------------------------------------------------
    # Low-level IPC
    # ------------------------------------------------------------------
    def _send(self, obj: dict) -> bool:
        proc = self._proc
        if proc is None or proc.stdin is None:
            return False
        try:
            payload = json.dumps(obj, separators=(",", ":")) + "\n"
        except (TypeError, ValueError) as exc:
            logger.debug(f"TorquePlotter: json dump failed: {exc}")
            return False
        try:
            with self._stdin_lock:
                proc.stdin.write(payload)
                proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError, ValueError):
            # Subprocess has died or stdin was closed.
            self._proc = None
            return False


# ---------------------------------------------------------------------------
# Optional module-level singleton hooks for python console use.
# ---------------------------------------------------------------------------
_SINGLETONS: dict[str, TorquePlotter] = {}


def register_singleton(key: str, plotter: TorquePlotter) -> None:
    _SINGLETONS[key] = plotter


def get_singleton(key: str = "body") -> Optional[TorquePlotter]:
    return _SINGLETONS.get(key)
