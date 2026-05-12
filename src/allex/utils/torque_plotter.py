"""Realtime per-joint data plotter for the ALLEX digital twin.

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

logger = logging.getLogger("allex.data_plotter")


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
_DEFAULT_WINDOW_SECONDS = 20.0

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
class DataPlotter:
    """Subprocess-backed realtime per-joint data plotter (torque / position).

    Subplot 별로 ``channel`` 을 지정해 torque (N m) 또는 joint position (deg) 을
    그릴 수 있다. Channel 선택은 ``viz_config.json::torque_plot.subsets[*][*].channel``
    필드를 통해 group 단위로 한다. 한 subplot 은 한 channel.

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
        Physics step rate. Required — caller must pass the actual rate
        derived from ``physics_config.json::world.physics_dt`` (e.g.
        ``1.0 / get_world_settings()["physics_dt"]``). Used to compute the
        ``apply_step`` decim ratio (physics_hz / plot_hz). Must be > 0.
    window_s : float
        Visible time window (seconds).
    """

    def __init__(
        self,
        articulation,
        physics_hz: float,
        ff_provider: Optional[Callable[[], np.ndarray]] = None,
        subset: str = "body",
        window_s: float = _DEFAULT_WINDOW_SECONDS,
        plot_hz: float = 50.0,
        ff_manager=None,
        plot_mode: str = "rolling",
        save_on_exit: bool = False,
        real_provider: Optional[Callable[[], dict]] = None,
        y_lim: Optional[tuple] = None,
    ):
        if physics_hz is None or float(physics_hz) <= 0.0:
            raise ValueError(
                "DataPlotter: physics_hz must be > 0; pass "
                "1.0 / get_world_settings()['physics_dt']"
            )
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
        # None → autoscale (default). (ymin, ymax) → 고정 y축. plot_proc 에 그대로 전달.
        if y_lim is None:
            self._y_lim: Optional[tuple] = None
        else:
            try:
                a = float(y_lim[0])
                b = float(y_lim[1])
                self._y_lim = (a, b) if a < b else None
            except (TypeError, ValueError, IndexError):
                self._y_lim = None

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
        # Subset 에 pos channel 이 하나라도 있는지 — live mode 의 articulation
        # position 폴링 스킵 여부 결정 (없으면 매 step GPU→CPU sync 회피).
        self._has_pos_channel: bool = any(
            g.get("channel", "torque") == "pos" for g in self._groups
        )
        self._has_torque_channel: bool = any(
            g.get("channel", "torque") == "torque" for g in self._groups
        )
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
                f"DataPlotter({subset}): {len(missing)} joint(s) have no abbr "
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
        # CSV replay 의 position channel 용 — torque buffer 와 동일 패턴.
        # 단위는 rad (변환은 plot_proc 의 표시 직전에 한 번만).
        self._external_sim_pos: Optional[np.ndarray] = None
        self._external_real_pos_by_abbr: Optional[dict] = None

        # CSV replay 모드 — set_replay_mode(True) 면 apply_step 의 자동 push 끄고
        # push_replay_frame 으로 모든 CSV 샘플을 직접 받음.
        self._replay_mode: bool = False

        # Sim 데이터 소스 추적. "live" (articulation qfrc_actuator) / "csv_external"
        # (set_external_sim_tau buffer) / "csv_push" (push_replay_frame). 값이 바뀔
        # 때만 logger.info 로 한 번씩 알림 — step 마다 spam 안 함.
        self._last_sim_source: Optional[str] = None

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

    def _announce_sim_source(self, source: str) -> None:
        """Log sim data source transition (live ↔ CSV). Spam-free — only on change."""
        if source == self._last_sim_source:
            return
        self._last_sim_source = source
        label = {
            "live": "live articulation (qfrc_actuator)",
            "csv_external": "CSV replay (external sim_tau buffer)",
            "csv_push": "CSV replay (push_replay_frame)",
        }.get(source, source)
        logger.info(f"DataPlotter[{self._subset}] sim source = {label}")

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

    def set_external_sim_pos(self, arr) -> None:
        """CSV replay 동안 articulation 폴링 대신 사용할 (num_dof,) sim position (rad).

        rad → deg 변환은 plot_proc 의 표시 직전에 일괄 처리하므로 여기선 항상 rad.
        ``None`` clears the override.
        """
        if arr is None:
            self._external_sim_pos = None
            return
        self._external_sim_pos = np.asarray(arr, dtype=np.float32).reshape(-1)

    def set_external_real_pos_by_abbr(self, d) -> None:
        """CSV replay 동안 사용할 ``dict[abbr -> position(rad)]``. ``None`` clears."""
        self._external_real_pos_by_abbr = None if d is None else d

    # ------------------------------------------------------------------
    # Replay mode (called by CsvReplayer) — bypass apply_step's auto push,
    # accept direct frame pushes at native CSV sample rate.
    # ------------------------------------------------------------------
    def set_replay_mode(self, on: bool) -> None:
        """Replay 모드: ``apply_step`` 의 physics-tick 자동 push 비활성.

        On=True 인 동안 plotter 는 ``push_replay_frame`` 만 받아서 그린다.
        replayer 가 모든 CSV 샘플을 자기 t 로 직접 push 하므로 decim/throttle 무관.
        """
        self._replay_mode = bool(on)

    def reset_buffer(self) -> None:
        """Clear all deques in the subprocess.

        On replay stop→start the replayer pushes t from 0 again, so subprocess
        ``shared_t`` accumulates ``[old 0..T_old, new 0..T_new]``. ``set_data``
        then receives a non-monotonic t and matplotlib draws a back-line from
        the last to the first point, making one legend entry look like
        multiple traces. Call this once on restart to prevent that.
        """
        if not self.is_running():
            return
        self._send({"cmd": "reset"})
        # Reset Kit-side time/counter too so a stale value isn't used on
        # re-entry into apply_step (sim mode).
        self._sim_time = 0.0
        self._step_count = 0

    def push_replay_frame(
        self,
        t: float,
        sim_tau_full: Optional[np.ndarray],
        real_by_abbr: Optional[dict],
        dof_name_to_idx: dict,
        sim_pos_full: Optional[np.ndarray] = None,
        real_pos_by_abbr: Optional[dict] = None,
    ) -> None:
        """Replay 전용 직접 push — decim/step counter 무시.

        Parameters
        ----------
        t
            CSV 샘플의 절대 시간 (초). plot X 축에 그대로 사용.
        sim_tau_full
            (num_dof,) numpy 배열. None 이면 sim torque 채널 omit.
        real_by_abbr
            ``dict[abbr -> tau]``. None 이면 real torque 채널 omit.
        dof_name_to_idx
            sim_*_full 의 dof_name → idx 매핑. plotter 의 `_joint_order` 정렬용.
        sim_pos_full
            (num_dof,) numpy 배열, rad. None 이면 sim pos 채널 omit.
        real_pos_by_abbr
            ``dict[abbr -> pos(rad)]``. None 이면 real pos 채널 omit.
        """
        if not self.is_running():
            return
        self._announce_sim_source("csv_push")

        n_joints = len(self._joint_order)
        frame: dict = {"t": float(t)}

        def _build_sim_list(arr: np.ndarray) -> Optional[list]:
            out: list[float] = []
            for name in self._joint_order:
                idx = dof_name_to_idx.get(name)
                if idx is None or idx >= arr.size:
                    return None
                out.append(float(arr[idx]))
            return out

        # sim torque
        if sim_tau_full is not None and sim_tau_full.size > 0:
            sim_tau_list = _build_sim_list(sim_tau_full)
            if sim_tau_list is not None:
                frame["y_torque_sim"] = sim_tau_list

        # real torque
        if real_by_abbr is not None:
            frame["y_torque_real"] = [
                float(real_by_abbr.get(abbr, 0.0)) if abbr is not None else 0.0
                for abbr in self._joint_order_abbrs
            ]

        # sim pos (rad — plot_proc 가 deg 변환)
        if sim_pos_full is not None and sim_pos_full.size > 0:
            sim_pos_list = _build_sim_list(sim_pos_full)
            if sim_pos_list is not None:
                frame["y_pos_sim"] = sim_pos_list

        # real pos (rad)
        if real_pos_by_abbr is not None:
            nan = float("nan")
            frame["y_pos_real"] = [
                float(real_pos_by_abbr.get(abbr, nan)) if abbr is not None else nan
                for abbr in self._joint_order_abbrs
            ]

        # 새 키만 보내고 데이터가 아무것도 없으면 send skip — t 만 있는 빈 frame 무의미.
        if n_joints == 0 or len(frame) == 1:
            return
        self._send(frame)

    @property
    def plot_hz(self) -> float:
        return self._plot_hz

    # ------------------------------------------------------------------
    # Joint / group selection
    # ------------------------------------------------------------------
    @staticmethod
    def _select_joints(dof_names: list[str], subset: str) -> tuple[list[int], list[str]]:
        """Subset 으로 표시할 dof index / name 추출.

        ``viz_config.json::torque_plot.subsets[subset]`` 가 정의돼 있으면 그 group
        들에 나열된 joint 만 (articulation 의 dof_names 안에 있는 것에 한해) 반환.
        없으면 regex (_HAND_RE / _BODY_RE) 기반 fallback.
        """
        from ..config.viz_config import TORQUE_PLOT_SUBSETS
        groups = TORQUE_PLOT_SUBSETS.get(subset)
        if groups:
            wanted = {j for g in groups for j in g["joints"]}
            indices: list[int] = []
            names: list[str] = []
            for i, n in enumerate(dof_names):
                if n in wanted:
                    indices.append(i)
                    names.append(n)
            return indices, names
        pat = _HAND_RE if subset == "hand" else _BODY_RE
        indices = []
        names = []
        for i, n in enumerate(dof_names):
            if pat.search(n):
                indices.append(i)
                names.append(n)
        return indices, names

    @staticmethod
    def _build_groups(plot_names: list[str], subset: str) -> list[dict]:
        """Subset 의 joint 들을 subplot 단위 group 으로 분할.

        ``viz_config.json::torque_plot.subsets[subset]`` 가 있으면 거기 정의된
        ``[{"name": str, "joints": [...]}]`` 그대로 사용 (dof_names 에 없는
        joint 는 group 안에서 자동 drop, group 이 비면 그 group 자체도 drop).
        없으면 기존 hand → L/R 분할, body → arm L/R + Waist+Neck 분할로 fallback.
        """
        from ..config.viz_config import TORQUE_PLOT_SUBSETS
        cfg_groups = TORQUE_PLOT_SUBSETS.get(subset)
        if cfg_groups:
            present = set(plot_names)
            out: list[dict] = []
            for g in cfg_groups:
                jl = [j for j in g["joints"] if j in present]
                if not jl:
                    continue
                entry = {"name": g["name"], "joints": jl}
                # group.channel ("torque" | "pos") — start() 의 init_msg 분기 및
                # _has_pos_channel / _has_torque_channel 플래그 계산에 필수.
                if "channel" in g:
                    entry["channel"] = g["channel"]
                # group.y_lim 이 있으면 그대로 보존 — start() 에서 init 으로 전달.
                if "y_lim" in g:
                    entry["y_lim"] = list(g["y_lim"])
                # group.highlights (axvspan 띠) 도 동일 패턴으로 통과.
                if "highlights" in g:
                    entry["highlights"] = list(g["highlights"])
                out.append(entry)
            return out
        if subset == "hand":
            hand_L = [n for n in plot_names if n.startswith("L_")]
            hand_R = [n for n in plot_names if n.startswith("R_")]
            out = []
            if hand_L:
                out.append({"name": "Hand L", "joints": hand_L})
            if hand_R:
                out.append({"name": "Hand R", "joints": hand_R})
            return out
        # body fallback: arm L/R + Waist+Neck buckets.
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
            logger.info("DataPlotter already running")
            return True
        if not self._plot_indices or not self._groups:
            logger.warning(
                f"DataPlotter: no joints matched subset={self._subset}; not starting"
            )
            return False

        # 사용자가 plot 창을 X 버튼으로 닫은 경우 stop() 이 호출되지 않아 stale
        # state 가 남는다. 새 spawn 직전에 한 번 더 청소 — proc 가 이미 죽었을
        # 가능성을 is_running() 이 감지해 _proc=None 으로 정리하지만, 시간축
        # counter 들은 거기서 안 건드리므로 명시적으로 0 으로.
        self._step_count = 0
        self._sim_time = 0.0
        self._replay_mode = False
        self._external_sim_tau = None
        self._external_real_by_abbr = None
        self._external_sim_pos = None
        self._external_real_pos_by_abbr = None
        self._last_sim_source = None

        py = _resolve_plotter_python()
        if py is None:
            msg = (
                "[ALLEX][DataPlot] no external python found with "
                "(tkinter, matplotlib, numpy). Install: "
                "sudo apt install python3-tk python3-matplotlib python3-numpy, "
                "or set ALLEX_PLOTTER_PYTHON to a suitable interpreter."
            )
            print(msg)
            logger.error(msg)
            return False

        script = Path(__file__).parent / "plot_proc.py"
        if not script.is_file():
            logger.error(f"DataPlotter: proc script missing: {script}")
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
            logger.error(f"DataPlotter: Popen failed: {exc}")
            self._proc = None
            return False

        # 각 group 의 y_lim 을 resolve: group 자체 y_lim > subset 레벨 self._y_lim
        # > None (autoscale). plot_proc 는 group["y_lim"] 만 보면 되도록 미리 채움.
        fallback_ylim = list(self._y_lim) if self._y_lim is not None else None
        groups_resolved: list[dict] = []
        for g in self._groups:
            entry = {
                "name": g["name"],
                "joints": g["joints"],
                "channel": g.get("channel", "torque"),
            }
            if "y_lim" in g:
                entry["y_lim"] = list(g["y_lim"])
            elif fallback_ylim is not None and entry["channel"] == "torque":
                # subset-level y_lim (TORQUE_PLOT_Y_LIM_<subset>) 은 torque 단위 (N m)
                # 라서 pos channel 에 적용하면 의미가 어긋남. torque channel 에만 fallback.
                entry["y_lim"] = list(fallback_ylim)
            if "highlights" in g:
                entry["highlights"] = list(g["highlights"])
            groups_resolved.append(entry)

        init_msg = {
            "cmd": "init",
            "title": f"ALLEX Data ({self._subset}) [{self._plot_mode}]",
            "window_sec": self._window_s,
            "hz": self._physics_hz,
            "plot_mode": self._plot_mode,
            "save_on_exit": self._save_on_exit,
            "groups": groups_resolved,
            # Real second line (ROS2-fed). When False, subprocess skips dashed
            # real-line creation. Pos channel 의 live mode 에선 real provider 가
            # 없어도 CSV replay 에서 push 될 수 있어, 'has_real' 은 표시 가능성
            # 의미로만 사용 — 실제 데이터 유무는 채널별 frame key 가 결정.
            "has_real": self._real_provider is not None,
            # Legacy field — subset 전체 fallback (torque channel 만 적용).
            "y_lim": fallback_ylim,
        }
        if not self._send(init_msg):
            logger.error("DataPlotter: init handshake write failed")
            self.stop(timeout=1.0)
            return False

        logger.info(
            f"DataPlotter started (subset={self._subset}, "
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
                # SIGKILL 직후 OS 가 reap 할 시간 확보 — 곧바로 새 subprocess 가
                # spawn 되면 두 process 공존이 잠시 발생해서 plot UI 가 꼬일 수
                # 있음. 무한정 기다리지 않도록 5s 상한.
                try:
                    proc.wait(timeout=5.0)
                except Exception:
                    pass
        self._proc = None
        # 인스턴스 재사용 시 stale state 방지 — counter / 시간축 / replay flag /
        # 외부 override 버퍼 모두 리셋. 다음 start() 가 깨끗한 state 로 시작.
        self._step_count = 0
        self._sim_time = 0.0
        self._replay_mode = False
        self._external_sim_tau = None
        self._external_real_by_abbr = None
        self._external_sim_pos = None
        self._external_real_pos_by_abbr = None
        logger.info("DataPlotter stopped")

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
        # Replay 모드면 push_replay_frame 으로만 받음 — physics tick 자동 push 차단.
        if getattr(self, "_replay_mode", False):
            return
        # Throttle GPU→CPU reads to the plotter's render rate.
        if self._step_count % self._decim != 0:
            return

        num = self._num_dof
        tau_total: Optional[np.ndarray] = None
        # Pos channel 이 있는 subset 일 때만 채워지는 sim_pos (rad). torque 채널만
        # 인 경우 None — push_sample 에서 pos key 자동 omit.
        sim_pos: Optional[np.ndarray] = None

        # --- 외부 소스 (CSV replay 등) override 우선 ---
        ext_sim = self._external_sim_tau
        if ext_sim is not None and ext_sim.size >= num:
            tau_total = ext_sim[:num].astype(np.float32, copy=True)
            self._announce_sim_source("csv_external")

        if tau_total is None and self._has_torque_channel:
            self._announce_sim_source("live")
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
                    print(f"[ALLEX][DataPlot {self._subset}] "
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
                        print(f"[ALLEX][DataPlot {self._subset}] "
                              f"ff_manager.read_current failed: {exc}")

            # qfrc_actuator 가 아직 채워지지 않은 초기 몇 step 은 skip.
            if tau_pd_act is None or tau_pd_act.size < num:
                if self._step_count % (200 * self._decim) == 0:
                    size = None if tau_pd_act is None else int(tau_pd_act.size)
                    print(f"[ALLEX][DataPlot {self._subset}] qfrc_actuator unavailable "
                          f"(size={size}, need>={num}). "
                          f"Newton patch 가 빠졌거나 아직 build 전일 수 있음.")
                return

            tau_total = (tau_pd_act[:num] + tau_ff[:num]).astype(np.float32)

        # --- Live mode pos channel: articulation 으로부터 joint position 폴링 ---
        # _has_pos_channel 이 True 인 subset 에서만 실행. External override 가
        # 있으면 그쪽 우선.
        if self._has_pos_channel:
            ext_pos = self._external_sim_pos
            if ext_pos is not None and ext_pos.size >= num:
                sim_pos = ext_pos[:num].astype(np.float32, copy=True)
            else:
                try:
                    arr = self._articulation.get_joint_positions()
                    if arr is not None:
                        a = _to_np(arr)
                        if a is not None and a.ndim == 2:
                            a = a[0]
                        if a is not None and a.size >= num:
                            sim_pos = np.asarray(a, dtype=np.float32).reshape(-1)[:num]
                except Exception as exc:
                    if self._step_count % (200 * self._decim) == 0:
                        print(f"[ALLEX][DataPlot {self._subset}] "
                              f"get_joint_positions failed: {exc}")

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
        if self._has_torque_channel:
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
                        print(f"[ALLEX][DataPlot {self._subset}] "
                              f"real_provider failed: {exc}")
                    snap = None
                if isinstance(snap, dict) and snap:
                    # 누락된 abbr 는 NaN — 0 으로 채우면 ROS2 토픽이 아직 안 와 있을 때
                    # real line 이 y=0 에 고정되어 "원점에 붙은 직선" artifact 가 됨.
                    nan = float("nan")
                    tau_real_list = [
                        float(snap.get(abbr, nan)) if abbr is not None else nan
                        for abbr in self._joint_order_abbrs
                    ]
                else:
                    tau_real_list = None

        # --- real pos snapshot — live mode 에선 provider 없음. CSV replay 의
        # external override 만 사용.
        pos_real_list: Optional[list[float]] = None
        if self._has_pos_channel:
            ext_real_pos = self._external_real_pos_by_abbr
            if ext_real_pos is not None:
                nan = float("nan")
                pos_real_list = [
                    float(ext_real_pos.get(abbr, nan)) if abbr is not None else nan
                    for abbr in self._joint_order_abbrs
                ]

        self.push_sample(self._sim_time, tau_total, self._dof_name_to_idx,
                         tau_real_list=tau_real_list,
                         pos_full=sim_pos, pos_real_list=pos_real_list)

    def push_sample(
        self,
        t: float,
        tau_full: Optional[np.ndarray],
        dof_name_to_idx: dict,
        tau_real_list: Optional[list] = None,
        pos_full: Optional[np.ndarray] = None,
        pos_real_list: Optional[list] = None,
    ) -> None:
        """Send one sample frame to the subprocess.

        ``tau_full`` / ``pos_full`` are (num_dof,) numpy arrays. None 이면 해당
        채널 key 는 frame 에서 omit. pos 는 rad 단위 (변환은 plot_proc).

        ``tau_real_list`` / ``pos_real_list`` 는 ``_joint_order`` 길이의 list 또는
        None. None 이면 해당 real key omit.
        """
        if not self.is_running():
            return
        n_joints = len(self._joint_order)
        frame: dict = {"t": float(t)}

        def _build_sim_list(arr: np.ndarray) -> Optional[list]:
            out: list[float] = []
            for name in self._joint_order:
                idx = dof_name_to_idx.get(name)
                if idx is None or idx >= arr.size:
                    return None
                out.append(float(arr[idx]))
            return out

        if tau_full is not None and tau_full.size > 0:
            try:
                tau_list = _build_sim_list(tau_full)
            except Exception as exc:
                logger.debug(f"push_sample tau serialise failed: {exc}")
                tau_list = None
            if tau_list is not None:
                frame["y_torque_sim"] = tau_list

        if tau_real_list is not None and len(tau_real_list) == n_joints:
            frame["y_torque_real"] = [float(v) for v in tau_real_list]

        if pos_full is not None and pos_full.size > 0:
            try:
                pos_list = _build_sim_list(pos_full)
            except Exception as exc:
                logger.debug(f"push_sample pos serialise failed: {exc}")
                pos_list = None
            if pos_list is not None:
                frame["y_pos_sim"] = pos_list

        if pos_real_list is not None and len(pos_real_list) == n_joints:
            frame["y_pos_real"] = [float(v) for v in pos_real_list]

        if len(frame) == 1:
            return
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
            logger.debug(f"DataPlotter: json dump failed: {exc}")
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
_SINGLETONS: dict[str, DataPlotter] = {}


def register_singleton(key: str, plotter: DataPlotter) -> None:
    """동일 key 에 이미 다른 plotter 가 등록돼 있으면 먼저 stop 시킨 뒤 교체.

    Reset / articulation 재바인딩 시 이전 인스턴스의 subprocess 가 누수돼 Tk
    창이 남는 걸 방지.
    """
    prev = _SINGLETONS.get(key)
    if prev is not None and prev is not plotter:
        try:
            prev.stop()
        except Exception:
            pass
    _SINGLETONS[key] = plotter


def get_singleton(key: str = "body") -> Optional[DataPlotter]:
    return _SINGLETONS.get(key)


def clear_singletons() -> None:
    """Extension shutdown 시 호출 — 모든 등록된 plotter stop + dict clear."""
    for _name, plotter in list(_SINGLETONS.items()):
        try:
            plotter.stop()
        except Exception:
            pass
    _SINGLETONS.clear()
