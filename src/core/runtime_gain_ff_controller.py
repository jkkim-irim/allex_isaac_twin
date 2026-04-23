"""Runtime PD gain + feedforward torque injection for the ALLEX articulation.

Usage:
    ctrl = RuntimeGainFFController(articulation, dof_names)
    # per physics step, upstream resolves a GainFFSample and calls:
    ctrl.apply_step(sample)

No enable/disable flag — scenario decides whether to call ``apply_step`` based
on whether a ``GainFFCsvReader`` is ready. Column-level partial overrides are
honored via ``sample.*_mask``; DOFs without a corresponding CSV column are
left at their current values (for kp/kd) and receive 0 feedforward torque.

kp/kd changes are pushed via ``view.set_gains(..., joint_indices=<masked>)``
so unmasked DOFs keep their current drive gains. Identical consecutive
samples are skipped to avoid redundant solver notifications.

Feedforward torque path is a module-level constant (``_FF_PATH``):
  - "control.joint_f"  (default): writes directly into Newton's
    ``newton_stage.control.joint_f`` per step. Requires the articulation view
    to expose ``_newton_stage``. Verified in
    isaacsim.physics.newton/.../tensors/articulation_view.py:56-64 and
    newton/_src/sim/control.py:32-47.
  - "actuation_forces": uses ``ArticulationAction(joint_efforts=...)`` which
    routes to view.set_dof_actuation_forces.
Change the constant below to switch paths for Test E.
"""
from __future__ import annotations

import re

import numpy as np

from isaacsim.core.utils.types import ArticulationAction


# ---------------------------------------------------------------------------
# Feedforward torque application path. Swap this single line for Test E.
#   "control.joint_f"  | "actuation_forces"
# ---------------------------------------------------------------------------
_FF_PATH = "control.joint_f"


def _classify_ff_clamp(dof_names: list[str]) -> np.ndarray:
    """Per-group feedforward torque clamp (absolute value, N·m).

    Groups (regex on dof name):
      - hand fingers (Thumb|Index|Middle|Ring|Little): 5
      - wrist: 20
      - neck:  10
      - shoulder|elbow|waist: 100
      - other: 100
    """
    clamp = np.full(len(dof_names), 100.0, dtype=np.float32)
    finger_re = re.compile(r"Thumb|Index|Middle|Ring|Little", re.IGNORECASE)
    wrist_re = re.compile(r"Wrist", re.IGNORECASE)
    neck_re = re.compile(r"Neck", re.IGNORECASE)
    arm_re = re.compile(r"Shoulder|Elbow|Waist", re.IGNORECASE)
    for i, n in enumerate(dof_names):
        if finger_re.search(n):
            clamp[i] = 5.0
        elif wrist_re.search(n):
            clamp[i] = 20.0
        elif neck_re.search(n):
            clamp[i] = 10.0
        elif arm_re.search(n):
            clamp[i] = 100.0
        else:
            clamp[i] = 100.0
    return clamp


class RuntimeGainFFController:
    def __init__(self, articulation, dof_names: list[str]):
        self._articulation = articulation
        self._view = getattr(articulation, "_articulation_view", None)
        self._dof_names = list(dof_names)
        self._num_dof = len(dof_names)
        self._ff_clamp = _classify_ff_clamp(self._dof_names)

        # Tracks the last sent (kp, kd) per DOF so repeated identical samples
        # don't trigger redundant set_gains / solver notifications. Indices
        # with mask=False in the incoming sample are ignored for comparison.
        self._last_kp: np.ndarray | None = None
        self._last_kd: np.ndarray | None = None

        # Lazily imported warp handle + one-shot warning flag.
        self._wp = None
        self._warned_ff_path_missing = False

        # Snapshot of the most recently applied feedforward torque vector
        # (post-clamp). Consumers such as the TorquePlotter read this via
        # ``get_last_applied_ff()`` when _FF_PATH == "control.joint_f" because
        # ``view.get_dof_actuation_forces()`` only reflects values written via
        # the actuation_forces path.
        self._last_applied_ff: np.ndarray = np.zeros(self._num_dof, dtype=np.float32)

    @property
    def ff_clamp(self) -> np.ndarray:
        return self._ff_clamp

    def get_last_applied_ff(self) -> np.ndarray:
        """Return a copy of the most recently applied (post-clamp) FF torque
        vector. Shape (num_dof,) float32. All zeros until the first apply_step
        call. Safe to call from any thread — returns a fresh copy.
        """
        return self._last_applied_ff.copy()

    # ------------------------------------------------------------------
    def apply_step(self, sample) -> None:
        """Apply one GainFFSample. ``None`` -> no-op."""
        if sample is None:
            return

        kp_mask = np.asarray(sample.kp_mask, dtype=bool).reshape(-1)
        kd_mask = np.asarray(sample.kd_mask, dtype=bool).reshape(-1)
        ff_mask = np.asarray(sample.ff_mask, dtype=bool).reshape(-1)

        # ---- kp/kd: partial override via joint_indices --------------
        active_mask = kp_mask | kd_mask
        if active_mask.any():
            kp_new = np.asarray(sample.kp, dtype=np.float32).reshape(-1)
            kd_new = np.asarray(sample.kd, dtype=np.float32).reshape(-1)

            # Suppress no-op: compare only the currently-masked DOFs against
            # the last sent values (for the same DOFs).
            changed = True
            if self._last_kp is not None and self._last_kd is not None:
                same_kp = np.allclose(
                    kp_new[kp_mask], self._last_kp[kp_mask], rtol=1e-5, atol=1e-6
                ) if kp_mask.any() else True
                same_kd = np.allclose(
                    kd_new[kd_mask], self._last_kd[kd_mask], rtol=1e-5, atol=1e-6
                ) if kd_mask.any() else True
                changed = not (same_kp and same_kd)

            if changed:
                self._apply_gains_partial(kp_new, kd_new, kp_mask, kd_mask)
                # Update the last-sent cache for masked DOFs only.
                if self._last_kp is None:
                    self._last_kp = np.zeros(self._num_dof, dtype=np.float32)
                if self._last_kd is None:
                    self._last_kd = np.zeros(self._num_dof, dtype=np.float32)
                self._last_kp[kp_mask] = kp_new[kp_mask]
                self._last_kd[kd_mask] = kd_new[kd_mask]

        # ---- FF torque: masked-off positions forced to zero ---------
        tau = np.zeros(self._num_dof, dtype=np.float32)
        if ff_mask.any():
            ff_new = np.asarray(sample.tau_ff, dtype=np.float32).reshape(-1)
            tau[ff_mask] = np.clip(
                ff_new[ff_mask],
                -self._ff_clamp[ff_mask],
                +self._ff_clamp[ff_mask],
            )
        # Always write — control.joint_f is persistent across steps, so an
        # all-zero write is required when ff_mask has no active DOFs to
        # prevent stale torques from lingering. The assign below fully
        # overwrites the entire joint_f array.
        self._apply_ff(tau)

    # ------------------------------------------------------------------
    def _apply_gains_partial(
        self,
        kp: np.ndarray,
        kd: np.ndarray,
        kp_mask: np.ndarray,
        kd_mask: np.ndarray,
    ) -> None:
        """Call ``view.set_gains`` on the masked DOFs only.

        When kp_mask and kd_mask differ, we split into two calls: one updating
        only kp for ``kp_mask & ~kd_mask`` DOFs (kds=None), one updating only
        kd for ``kd_mask & ~kp_mask`` DOFs (kps=None), and one updating both
        for ``kp_mask & kd_mask`` DOFs.
        Reference (kps/kds/joint_indices semantics):
        isaacsim.core.prims/python/impl/articulation.py:2884-2966.
        """
        view = self._view
        if view is None:
            return

        try:
            import torch as _torch
            _to_tensor = lambda arr: _torch.tensor(arr)
        except Exception:
            _to_tensor = lambda arr: arr

        def _call(mask: np.ndarray, send_kp: bool, send_kd: bool):
            if not mask.any():
                return
            joint_idx = np.where(mask)[0].astype(np.int64)
            sub_kp = kp[mask].reshape(1, -1).astype(np.float32) if send_kp else None
            sub_kd = kd[mask].reshape(1, -1).astype(np.float32) if send_kd else None
            try:
                view.set_gains(
                    kps=_to_tensor(sub_kp) if sub_kp is not None else None,
                    kds=_to_tensor(sub_kd) if sub_kd is not None else None,
                    joint_indices=joint_idx,
                )
            except Exception as exc:
                print(f"[ALLEX][GainFF] set_gains(joint_indices=...) failed: {exc}")

        both = kp_mask & kd_mask
        only_kp = kp_mask & ~kd_mask
        only_kd = kd_mask & ~kp_mask
        _call(both, send_kp=True, send_kd=True)
        _call(only_kp, send_kp=True, send_kd=False)
        _call(only_kd, send_kp=False, send_kd=True)

    # ------------------------------------------------------------------
    def _apply_ff(self, tau: np.ndarray) -> None:
        # Snapshot for observers (e.g. TorquePlotter). np.ndarray assignment
        # of a same-shape float32 buffer is atomic enough for plot consumers
        # that only need "recent enough" values; they copy before use.
        try:
            self._last_applied_ff[:] = tau.astype(np.float32, copy=False)
        except Exception:
            # Shape mismatch (e.g. num_dof changed after reinit) — reallocate.
            self._last_applied_ff = tau.astype(np.float32, copy=True)

        if _FF_PATH == "actuation_forces":
            self._apply_ff_via_action(tau)
        else:
            # Default: write directly to newton_stage.control.joint_f.
            self._apply_ff_via_joint_f(tau)

    def _apply_ff_via_action(self, tau: np.ndarray) -> None:
        try:
            action = ArticulationAction(joint_efforts=tau)
            self._articulation.apply_action(action)
        except Exception as exc:
            if not self._warned_ff_path_missing:
                print(f"[ALLEX][GainFF] apply_action(joint_efforts) failed: {exc}")
                self._warned_ff_path_missing = True

    def _apply_ff_via_joint_f(self, tau: np.ndarray) -> None:
        view = self._view
        if view is None:
            return
        stage = getattr(view, "_newton_stage", None)
        if stage is None:
            if not self._warned_ff_path_missing:
                print("[ALLEX][GainFF] view._newton_stage is None; cannot write control.joint_f")
                self._warned_ff_path_missing = True
            return
        control = getattr(stage, "control", None)
        joint_f = getattr(control, "joint_f", None) if control is not None else None
        if joint_f is None:
            if not self._warned_ff_path_missing:
                print("[ALLEX][GainFF] control.joint_f is None; model likely not finalized yet")
                self._warned_ff_path_missing = True
            return

        if self._wp is None:
            try:
                import warp as _wp
                self._wp = _wp
            except Exception as exc:
                print(f"[ALLEX][GainFF] warp import failed: {exc}")
                return

        device_str = getattr(stage, "device_str", None)
        try:
            src = self._wp.array(tau.astype(np.float32), dtype=self._wp.float32, device=device_str)
            # joint_f shape is (joint_dof_count,). The articulation-local DOF
            # order from dof_names matches the articulation's axis ordering;
            # for a single-articulation stage this maps 1:1 to joint_f.
            if joint_f.shape[0] == src.shape[0]:
                joint_f.assign(src)
            else:
                # Partial assign: copy into first num_dof entries of joint_f.
                host = joint_f.numpy().copy()
                host[: src.shape[0]] = tau.astype(np.float32)
                joint_f.assign(self._wp.array(host, dtype=self._wp.float32, device=device_str))
        except Exception as exc:
            if not self._warned_ff_path_missing:
                print(f"[ALLEX][GainFF] control.joint_f assign failed: {exc}")
                self._warned_ff_path_missing = True
