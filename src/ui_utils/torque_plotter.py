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
        ``RuntimeGainFFController.get_last_applied_ff``.
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
    ):
        self._articulation = articulation
        self._ff_provider = ff_provider
        self._subset = subset
        self._physics_hz = float(physics_hz)
        self._window_s = float(window_s)
        self._plot_hz = float(plot_hz)

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

        # Subprocess state.
        self._proc: Optional[subprocess.Popen] = None
        self._stdin_lock = threading.Lock()
        self._sim_time: float = 0.0

        # Decimate physics step → plotter sampling. FuncAnimation renders at
        # 20 Hz so much higher than ~50 Hz is wasted GPU→CPU sync.
        self._decim = self._compute_decim(self._plot_hz)
        self._step_count = 0

    def _compute_decim(self, plot_hz: float) -> int:
        hz = max(1.0, float(plot_hz))
        return max(1, int(round(self._physics_hz / hz)))

    def set_plot_hz(self, plot_hz: float) -> None:
        """Change sampling rate at runtime. Effective immediately; no restart."""
        self._plot_hz = max(1.0, float(plot_hz))
        self._decim = self._compute_decim(self._plot_hz)

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
            "title": f"ALLEX Torque ({self._subset})",
            "window_sec": self._window_s,
            "hz": self._physics_hz,
            "y_label": "torque [N m]",
            "groups": self._groups,
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

        Recomputes τ_total, then pushes a sample frame to the subprocess
        if it is alive. Silent no-op when subprocess isn't running — this
        lets ``scenario._push_torque_sample`` call both body/hand plotters
        unconditionally.
        """
        self._step_count += 1
        if not self.is_running():
            return
        if not self._plot_indices:
            return
        # Throttle GPU→CPU reads to the plotter's render rate.
        if self._step_count % self._decim != 0:
            return

        try:
            view = getattr(self._articulation, "_articulation_view", None)
            if view is None:
                if self._step_count % (200 * self._decim) == 0:
                    print(f"[ALLEX][TorquePlot {self._subset}] drop: no _articulation_view")
                return

            pos = _to_np(view.get_joint_positions())
            vel = _to_np(view.get_joint_velocities())
            kps, kds = view.get_gains()
            kp_vec = _to_np(kps)
            kd_vec = _to_np(kds)
            actions = view.get_applied_actions(clone=False)
            pos_tgt = _to_np(actions.joint_positions) if actions is not None else None
            vel_tgt = _to_np(actions.joint_velocities) if actions is not None else None
        except Exception as exc:
            if self._step_count % (200 * self._decim) == 0:
                print(f"[ALLEX][TorquePlot {self._subset}] read failed: {exc}")
            return

        if pos is None or kp_vec is None or kd_vec is None or pos_tgt is None:
            if self._step_count % 200 == 0:
                sizes = (
                    None if pos is None else pos.size,
                    None if kp_vec is None else kp_vec.size,
                    None if kd_vec is None else kd_vec.size,
                    None if pos_tgt is None else pos_tgt.size,
                )
                print(f"[ALLEX][TorquePlot {self._subset}] drop: None input "
                      f"(pos,kp,kd,pos_tgt)={sizes}")
            return
        if vel is None:
            vel = np.zeros_like(pos)
        if vel_tgt is None:
            vel_tgt = np.zeros_like(pos)

        num = self._num_dof
        if (
            pos.size < num or vel.size < num or kp_vec.size < num or kd_vec.size < num
            or pos_tgt.size < num
        ):
            return
        pos = pos[:num]
        vel = vel[:num]
        kp_vec = kp_vec[:num]
        kd_vec = kd_vec[:num]
        pos_tgt = pos_tgt[:num]
        vel_tgt = vel_tgt[:num] if vel_tgt.size >= num else np.zeros(num, dtype=np.float32)

        tau_pd = kp_vec * (pos_tgt - pos) + kd_vec * (vel_tgt - vel)

        tau_ff = np.zeros(num, dtype=np.float32)
        if self._ff_provider is not None:
            try:
                ff = self._ff_provider()
                if ff is not None:
                    ff = np.asarray(ff, dtype=np.float32).reshape(-1)
                    if ff.size >= num:
                        tau_ff = ff[:num]
            except Exception as exc:
                logger.debug(f"ff_provider failed: {exc}")

        tau_total = (tau_pd + tau_ff).astype(np.float32)

        effective_dt = (float(dt) if dt else (1.0 / self._physics_hz)) * self._decim
        self._sim_time += effective_dt
        self.push_sample(self._sim_time, tau_total, self._dof_name_to_idx)

    def push_sample(
        self,
        t: float,
        tau_full: np.ndarray,
        dof_name_to_idx: dict,
    ) -> None:
        """Send one sample frame to the subprocess.

        ``tau_full`` is expected to be shape (num_dof,). The values are
        picked in ``self._joint_order`` order (which matches the init
        handshake). Missing joints are skipped; the subprocess enforces
        length equality and drops mismatched frames.
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
        self._send({"t": float(t), "y": tau_list})

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
