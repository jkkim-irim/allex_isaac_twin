"""Feedforward torque manager — Newton control.joint_f write & tracking.

Newton MuJoCo backend 는 현재 `mj_data.qfrc_applied` 를 state 로 expose
하지 않음 (허용 화이트리스트: body_parent_f / body_qdd /
mujoco:qfrc_actuator 만). 따라서 `control.joint_f.assign(...)` 으로
**쓴 FF torque 를 sim 쪽에서 되읽는 공식 API 가 없음**.

이 매니저는 extension 에서 FF 를 주입할 때 **쓴 값 자체를 보관** 해서
나중에 visualizer/logger 등이 조회할 수 있게 해 준다. control.joint_f 는
입력을 그대로 qfrc_applied 로 복사만 하는 pass-through 구조라
"쓴 값 = 실제 적용값" 으로 간주해도 무방하다 (solver_mujoco.py:3239 참고).

자세한 배경은 docs/newton_force_access.md 참고.
"""
from __future__ import annotations

import numpy as np


class FeedforwardTorqueManager:
    """Write FF torque to Newton's control.joint_f and remember the last
    applied vector.

    Usage (scenario 또는 controller 쪽):
        ff = FeedforwardTorqueManager(num_dof=60)
        ff.apply(tau_array)             # 매 physics step 전에
        tau_cur = ff.get_last()         # 시각화/로깅 측에서 언제든 조회

    ``num_dof`` 미지정 시 첫 ``apply()`` 호출 때 자동 감지.
    """

    def __init__(self, num_dof: int = 0):
        self._num_dof = int(num_dof)
        self._last = (
            np.zeros(num_dof, dtype=np.float32) if num_dof > 0 else None
        )
        self._wp = None  # lazy warp import
        self._warned_write_fail = False

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    @property
    def num_dof(self) -> int:
        return self._num_dof

    def set_num_dof(self, num_dof: int) -> None:
        """(Re)allocate the internal buffer to match the articulation size."""
        num_dof = int(num_dof)
        if num_dof == self._num_dof and self._last is not None:
            return
        self._num_dof = num_dof
        self._last = np.zeros(num_dof, dtype=np.float32)

    # ------------------------------------------------------------------
    # Write / read
    # ------------------------------------------------------------------
    def apply(self, tau) -> bool:
        """Write ``tau`` into Newton's ``control.joint_f`` and store a copy.

        Args:
            tau: numpy array-like, shape (N,). Truncated/padded to the
                 articulation DOF count if mismatched.

        Returns:
            True if Newton stage write succeeded, False otherwise.
            The internal ``_last`` snapshot is updated regardless (so
            observers can at least see the user's intent even if the
            write to Newton failed).
        """
        arr = np.asarray(tau, dtype=np.float32).reshape(-1)
        if self._num_dof <= 0:
            self.set_num_dof(arr.shape[0])
        # Always update the internal snapshot (observer-facing).
        if arr.shape[0] == self._num_dof:
            self._last[:] = arr
        else:
            buf = np.zeros(self._num_dof, dtype=np.float32)
            n = min(arr.shape[0], self._num_dof)
            buf[:n] = arr[:n]
            self._last[:] = buf
        return self._write_to_newton(self._last)

    def get_last(self) -> np.ndarray | None:
        """Copy of the most recently applied FF torque vector **by this
        manager**, or None if never applied. Safe to call from any thread.

        ⚠ 주의: 외부 코드 경로가 ``control.joint_f`` 에 직접 써도 여기엔
        반영 안 됨. Ground truth 가 필요하면 :meth:`read_current` 를 쓸 것.
        """
        if self._last is None:
            return None
        return self._last.copy()

    def read_current(self) -> np.ndarray | None:
        """Read back ``control.joint_f`` from Newton directly (ground truth).

        모든 writer (this manager, 외부 script 등) 가 써둔 **최종 FF torque
        벡터** 를 반환. solver 가 실제 pass-through 할 값과 동일.

        Returns: (num_dof,) float32 ndarray, 실패 시 None.
        """
        try:
            from isaacsim.physics.newton import acquire_stage
        except Exception:
            return None
        st = acquire_stage()
        if st is None:
            return None
        ctrl = getattr(st, "control", None)
        if ctrl is None:
            return None
        jf = getattr(ctrl, "joint_f", None)
        if jf is None:
            return None
        try:
            return np.asarray(jf.numpy(), dtype=np.float32).reshape(-1)
        except Exception:
            return None

    def clear(self) -> bool:
        """Write zeros and update snapshot. Useful when stopping playback
        so residual FF torque doesn't linger in ``control.joint_f``.
        """
        if self._num_dof <= 0:
            return False
        zeros = np.zeros(self._num_dof, dtype=np.float32)
        self._last[:] = zeros
        return self._write_to_newton(zeros)

    # ------------------------------------------------------------------
    # Internal: Newton write
    # ------------------------------------------------------------------
    def _write_to_newton(self, tau: np.ndarray) -> bool:
        try:
            from isaacsim.physics.newton import acquire_stage
        except Exception as exc:
            self._warn_once(f"acquire_stage import failed: {exc}")
            return False

        st = acquire_stage()
        if st is None:
            return False
        control = getattr(st, "control", None)
        if control is None:
            return False
        joint_f = getattr(control, "joint_f", None)
        if joint_f is None:
            return False

        if self._wp is None:
            try:
                import warp as _wp
                self._wp = _wp
            except Exception as exc:
                self._warn_once(f"warp import failed: {exc}")
                return False

        device_str = getattr(st, "device_str", None)
        try:
            if joint_f.shape[0] == tau.shape[0]:
                src = self._wp.array(tau, dtype=self._wp.float32, device=device_str)
                joint_f.assign(src)
            else:
                # Partial overwrite when articulation-local DOF count differs
                # from full joint_f length (multi-articulation scenes).
                host = joint_f.numpy().copy()
                n = min(tau.shape[0], host.shape[0])
                host[:n] = tau[:n]
                joint_f.assign(
                    self._wp.array(host, dtype=self._wp.float32, device=device_str)
                )
            return True
        except Exception as exc:
            self._warn_once(f"joint_f.assign failed: {exc}")
            return False

    def _warn_once(self, msg: str) -> None:
        if not self._warned_write_fail:
            print(f"[ALLEX][FF] {msg}")
            self._warned_write_fail = True
