"""Real-time sim state logger — Newton model/solver 로부터 PD 게인·토크를 CSV 로 저장.

검증 목적:
    - joint_target_ke/kd/effort_limit 이 q 에 따라 변하는지 (MotorStateMirror 동작 확인)
    - Waist_Lower_Pitch / Neck_Pitch 의 passive gravity (qfrc_gravcomp) 가 비-zero 인지
      (hardware spring 중력보상 모사 확인)
    - 나머지 관절의 qfrc_actuator 에 gravity compensation 이 합산되는지

출력 파일 (data/showcase/showcase_sim_data_{ts}/ 내):
    joint_position.csv  — articulation.get_joint_positions() per DOF
    joint_torque.csv    — qfrc_actuator per DOF (Newton DOF 순서 직접 인덱싱)
    passive_torque.csv  — qfrc_gravcomp for Waist_Lower_Pitch + Neck_Pitch
    kp_gain.csv         — joint_target_ke per DOF
    kd_gain.csv         — joint_target_kd per DOF
    torque_limit.csv    — joint_effort_limit per DOF
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime

import numpy as np


_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data")
)
_SHOWCASE_DIR = os.path.join(_DATA_DIR, "showcase")
_JOINT_CONFIG = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "joint_config.json")
)

PASSIVE_JOINTS: tuple[str, ...] = ("Waist_Lower_Pitch_Joint", "Neck_Pitch_Joint")


class SimStateLogger:
    """Newton 시뮬레이션 상태 logger.

    Args:
        articulation: ALLEX articulation (dof_names + get_joint_positions 제공).
        model:        Newton Model (joint_target_ke/kd/effort_limit).
        solver:       Newton MuJoCo solver (mjw_data 제공).
        log_every:    N 스텝마다 1회 캡처 (기본 1 = 1000Hz at 1000Hz sim).
    """

    def __init__(self, articulation, model, solver, log_every: int = 1):
        self._articulation = articulation
        self._model = model
        self._solver = solver
        self._log_every = max(1, log_every)

        dof_names = list(getattr(articulation, "dof_names", []) or [])
        self._dof_names = dof_names
        self._n_dofs = len(dof_names)

        name_to_idx = {n: i for i, n in enumerate(dof_names)}

        # Passive joint DOF indices in Newton ordering (hardware spring joints)
        self._passive_idx: list[int] = [
            name_to_idx[n] for n in PASSIVE_JOINTS if n in name_to_idx
        ]
        self._passive_names: list[str] = [n for n in PASSIVE_JOINTS if n in name_to_idx]

        # Actuated joint indices — joints with actual motors (from nominal_motor_gains)
        # Excludes passive/phantom joints (Waist_Upper_Pitch, DIP/IP follower, etc.)
        try:
            with open(_JOINT_CONFIG, "r", encoding="utf-8") as f:
                jcfg = json.load(f)
            motor_names = set(k for k in jcfg.get("nominal_motor_gains", {}) if k != "_notes")
        except Exception:
            motor_names = set()
        # Preserve articulation DOF order, keep only those with motors
        self._torque_names: list[str] = [n for n in dof_names if n in motor_names]
        self._torque_idx: list[int] = [name_to_idx[n] for n in self._torque_names]

        self._buf: dict[str, list[np.ndarray]] = {
            "joint_position": [],
            "joint_torque": [],
            "passive_torque": [],
            "kp_gain": [],
            "kd_gain": [],
            "torque_limit": [],
        }
        self._active = False
        self._step_count = 0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #
    def start(self, log_every: int | None = None, physics_hz: float = 1000.0,
              start_offset_steps: int = 0) -> None:
        """로깅 시작. log_every 스텝마다 캡처.

        Args:
            log_every: N 스텝마다 1회 캡처 (None → 기존 값 유지).
            physics_hz: 물리 step rate (1행당 경과 시간 계산용).
            start_offset_steps: ``start()`` 호출 후 처음 N step 동안은
                ``step()`` 이 호출돼도 캡처하지 않고 skip. 외부 데이터
                (e.g. rosbag) 와 시작 시점을 정렬할 때 사용. 0 = 즉시 시작.
        """
        if log_every is not None:
            self._log_every = max(1, log_every)
        self._dt = self._log_every / physics_hz  # 캡처 1행당 경과 시간 (s)
        for lst in self._buf.values():
            lst.clear()
        self._time_buf: list[float] = []
        self._current_time = 0.0
        self._step_count = 0
        self._offset_remaining = max(0, int(start_offset_steps))
        self._active = True
        offset_msg = (f", offset={self._offset_remaining} steps "
                      f"({self._offset_remaining/physics_hz:.2f}s)") if self._offset_remaining else ""
        print(
            f"[SimStateLogger] started — log_every={self._log_every}, "
            f"physics_hz={physics_hz:.0f}, dt={self._dt*1000:.2f} ms/row{offset_msg}"
        )

    def stop(self) -> str:
        """로깅 중단 후 CSV 저장. 출력 디렉터리 경로 반환."""
        self._active = False
        return self._write()

    def is_active(self) -> bool:
        return self._active

    def step(self) -> None:
        """매 physics step 에 호출 (pre_step_fn 또는 post-step 위치 무관)."""
        if not self._active:
            return
        # Drain offset window first (for time-aligning with external data).
        if self._offset_remaining > 0:
            self._offset_remaining -= 1
            return
        self._step_count += 1
        if self._step_count % self._log_every != 0:
            return
        try:
            self._capture()
        except Exception as exc:
            print(f"[SimStateLogger] capture error: {exc}")
            self._active = False  # 에러 시 자동 중단

    # ------------------------------------------------------------------ #
    # Capture                                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _slice0(arr) -> np.ndarray:
        """warp array (nworld, nv) → numpy 1-D (nv,)."""
        np_arr = arr.numpy()
        return np_arr[0] if np_arr.ndim == 2 else np_arr

    def _capture(self) -> None:
        import warp as wp
        wp.synchronize()

        self._time_buf.append(self._current_time)
        self._current_time += self._dt

        # Joint positions — actuated joints only
        try:
            qpos_raw = self._articulation.get_joint_positions()
            if hasattr(qpos_raw, "cpu"):
                qpos_raw = qpos_raw.cpu().numpy()
            elif hasattr(qpos_raw, "numpy"):
                qpos_raw = qpos_raw.numpy()
            qpos = np.asarray(qpos_raw).reshape(-1).astype(np.float32)
            pos_out = np.array([qpos[i] if i < len(qpos) else 0.0
                                for i in self._torque_idx], dtype=np.float32)
        except Exception:
            pos_out = np.zeros(len(self._torque_idx), dtype=np.float32)
        self._buf["joint_position"].append(pos_out)

        # PD gains / effort limit — actuated joints only
        ke  = self._model.joint_target_ke.numpy()
        kd  = self._model.joint_target_kd.numpy()
        eff = self._model.joint_effort_limit.numpy()
        self._buf["kp_gain"].append(np.array([ke[i] for i in self._torque_idx], dtype=np.float32))
        self._buf["kd_gain"].append(np.array([kd[i] for i in self._torque_idx], dtype=np.float32))
        self._buf["torque_limit"].append(np.array([eff[i] for i in self._torque_idx], dtype=np.float32))

        # Torques from mjw_data — direct Newton DOF indexing (same as old showcase_logger)
        # mjw_data.qfrc_actuator shape: (nworld, nv) where nv == Newton n_dofs for ALLEX
        mjw_data = getattr(self._solver, "mjw_data", None)
        n_torque = len(self._torque_idx)
        if mjw_data is not None:
            try:
                act = self._slice0(mjw_data.qfrc_actuator).astype(np.float32)
                act_out = np.array([act[i] if i < len(act) else 0.0
                                    for i in self._torque_idx], dtype=np.float32)
            except Exception:
                act_out = np.zeros(n_torque, dtype=np.float32)
            try:
                grav = self._slice0(mjw_data.qfrc_gravcomp).astype(np.float32)
            except Exception:
                grav = None
        else:
            act_out = np.zeros(n_torque, dtype=np.float32)
            grav = None

        self._buf["joint_torque"].append(act_out)
        if grav is not None and self._passive_idx:
            passive = np.array([grav[i] if i < len(grav) else 0.0
                                for i in self._passive_idx], dtype=np.float32)
        else:
            passive = np.zeros(len(self._passive_names), dtype=np.float32)
        self._buf["passive_torque"].append(passive)

    # ------------------------------------------------------------------ #
    # CSV write                                                            #
    # ------------------------------------------------------------------ #
    def _write(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(_SHOWCASE_DIR, f"showcase_sim_data_{ts}")
        os.makedirs(out_dir, exist_ok=True)

        n = len(self._buf["kp_gain"])
        if n == 0:
            print("[SimStateLogger] no data captured — nothing written")
            return out_dir

        time_arr = np.array(self._time_buf, dtype=np.float64)  # [n_steps]

        pos_cols = ["pos_" + n.split("/")[-1] for n in self._torque_names]

        specs: list[tuple[str, str, list[str]]] = [
            ("joint_position.csv", "joint_position", pos_cols),
            ("joint_torque.csv",   "joint_torque",   self._torque_names),
            ("passive_torque.csv", "passive_torque", self._passive_names),
            ("kp_gain.csv",        "kp_gain",        self._torque_names),
            ("kd_gain.csv",        "kd_gain",        self._torque_names),
            ("torque_limit.csv",   "torque_limit",   self._torque_names),
        ]
        for fname, key, cols in specs:
            data = self._buf[key]
            if not data or not cols:
                continue
            fpath = os.path.join(out_dir, fname)
            arr = np.array(data)  # [n_steps, n_cols]
            with open(fpath, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time"] + cols)
                for t, row in zip(time_arr, arr):
                    w.writerow([f"{t:.4f}"] + [f"{v:.6f}" for v in row])

        print(f"[SimStateLogger] {n} samples saved → {out_dir}")
        return out_dir
