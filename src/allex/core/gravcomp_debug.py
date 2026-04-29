"""Periodic torque debug printer for verifying MuJoCo Warp gravity compensation.

With `mjc:actuatorgravcomp = false` on every relevant joint (passive routing,
authored in `asset/ALLEX/ALLEX.usd`), the three quantities are cleanly split:

    qfrc_actuator  =  PD controller output only
    qfrc_gravcomp  =  feedforward gravity compensation only
    sum            =  what the joint actually feels (effort budget)

For mjc:actuatorgravcomp=true (actuator routing) `qfrc_actuator` already
contains both PD and gravcomp; the printer would still work but PD-only would
require subtracting `qfrc_gravcomp` from `qfrc_actuator`.
"""
from __future__ import annotations


_DEFAULT_JOINT = "R_Shoulder_Pitch_Joint"


class GravcompTorqueProbe:
    """Print PD torque, gravity comp torque, and their sum for one joint.

    DOF index is resolved on first emission against `articulation.dof_names` and
    cached. Reads are direct from `mjw_data.qfrc_*` arrays — same path the
    showcase logger uses to bypass Newton's unimplemented
    `get_dof_projected_joint_forces`.
    """

    def __init__(self, joint_name: str = _DEFAULT_JOINT, period: int = 200):
        self._joint_name = joint_name
        self._period = max(1, int(period))
        self._step = 0
        self._dof_idx: int | None = None
        self._warned_no_dof = False
        self._warned_no_data = False

    def step(self, articulation) -> None:
        self._step += 1
        if self._step % self._period != 0:
            return

        if self._dof_idx is None:
            try:
                names = list(getattr(articulation, "dof_names", []) or [])
                self._dof_idx = names.index(self._joint_name)
            except (ValueError, TypeError):
                if not self._warned_no_dof:
                    print(f"[ALLEX][Gravcomp] joint '{self._joint_name}' not in "
                          f"articulation.dof_names; probe disabled")
                    self._warned_no_dof = True
                return

        try:
            from isaacsim.physics.newton import acquire_stage
            stage = acquire_stage()
            solver = getattr(stage, "solver", None) if stage is not None else None
            mjw_data = getattr(solver, "mjw_data", None) if solver is not None else None
            if mjw_data is None:
                return
            qact_arr = getattr(mjw_data, "qfrc_actuator", None)
            qgrav_arr = getattr(mjw_data, "qfrc_gravcomp", None)
            if qact_arr is None or qgrav_arr is None:
                if not self._warned_no_data:
                    print("[ALLEX][Gravcomp] qfrc_actuator / qfrc_gravcomp not "
                          "available on mjw_data")
                    self._warned_no_data = True
                return
            qact = float(qact_arr.numpy()[0, self._dof_idx])
            qgrav = float(qgrav_arr.numpy()[0, self._dof_idx])
        except Exception as exc:
            if not self._warned_no_data:
                print(f"[ALLEX][Gravcomp] read failed: {exc}")
                self._warned_no_data = True
            return

        print(
            f"[ALLEX][Gravcomp][{self._joint_name}] "
            f"PD={qact:+8.3f}  Grav={qgrav:+8.3f}  Sum={qact + qgrav:+8.3f}  N·m"
        )
