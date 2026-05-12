"""Real-time sim state logger — Newton model/solver 로부터 PD 게인·토크를 CSV 로 저장.

검증 목적:
    - joint_target_ke/kd/effort_limit 이 q 에 따라 변하는지 (MotorStateMirror 동작 확인)
    - Waist_Lower_Pitch / Neck_Pitch 의 passive gravity (qfrc_gravcomp) 가 비-zero 인지
      (hardware spring 중력보상 모사 확인)
    - 나머지 관절의 qfrc_actuator 에 gravity compensation 이 합산되는지

출력 파일 (data/showcase/showcase_sim_data_{ts}/ 내):
    joint_position.csv  — articulation.get_joint_positions() per DOF
    joint_torque.csv    — qfrc_actuator per DOF (MuJoCo PD 출력; ke=kd=0 이면 0 또는 grav)
    qfrc_applied.csv    — qfrc_applied per DOF (= MotorStateMirror 의 joint_f, real
                          rosbag joint_torque 와 비교용)
    qfrc_gravcomp.csv   — qfrc_gravcomp per DOF (MuJoCo passive gravcomp, 모든 48 motor)
    passive_torque.csv  — qfrc_gravcomp for Waist_Lower_Pitch + Neck_Pitch (legacy 호환)
    kp_gain.csv         — joint_target_ke per DOF
    kd_gain.csv         — joint_target_kd per DOF
    torque_limit.csv    — joint_effort_limit per DOF
    showcase.csv        — replay/showcase_reader.py 호환 단일 CSV. time + pos_<joint>
                          + torque_<joint>(=qfrc_actuator+qfrc_gravcomp)
                          + contact_pos / force / force_vec / aggregate_* columns.
                          contact_config.json::pairs / aggregate_groups[0] 기반.
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
_CONTACT_CONFIG = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "contact_config.json")
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

    def __init__(self, articulation, model, solver, log_every: int = 1, control=None):
        self._articulation = articulation
        self._model = model
        self._solver = solver
        self._control = control
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
            "joint_position_target": [],
            "joint_torque": [],
            "qfrc_applied": [],
            "qfrc_gravcomp": [],
            "passive_torque": [],
            "kp_gain": [],
            "kd_gain": [],
            "torque_limit": [],
            "K_m_host": [],   # MotorStateMirror per-step ramped motor-side K_m
            "Kv_m_host": [],
            "trq_m_host": [],
            # Contact extraction (showcase.csv) — lazily resolved against
            # config/contact_config.json + Newton model.shape_label.
            "contact_pair_pos":   [],   # (n_pairs, 3)
            "contact_pair_scalar":[],   # (n_pairs,) signed if flip_direction
            "contact_pair_vec":   [],   # (n_pairs, 3)
            "contact_agg_origin": [],   # (3,) zeros if no aggregate group
            "contact_agg_total":  [],   # scalar
            "contact_agg_normal": [],   # (3,)
        }
        # contact column specs — resolved lazily on first _capture() because
        # Newton model.shape_label may not be populated at logger init.
        self._pair_specs: list[dict | None] | None = None
        self._aggregate_spec: dict | None = None
        self._contact_resolve_warned = False
        # Late-bound: scenario sets _scenario_ref so we can dynamically read
        # scenario._motor_mirror (lazily built on first physics step).
        self._scenario_ref = None
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

    def _resolve_contact_columns(self) -> None:
        """Build self._pair_specs / self._aggregate_spec from contact_config.json
        + Newton model.shape_label. Idempotent; safe to call repeatedly.

        Sets self._pair_specs = [] when config or shape_label missing (contact
        capture becomes a no-op while logger still works for joint CSVs).
        """
        if self._pair_specs is not None:
            return
        shape_label = list(getattr(self._model, "shape_label", None) or [])
        if not shape_label:
            if not self._contact_resolve_warned:
                print("[SimStateLogger] model.shape_label not ready; contact "
                      "columns will stay empty in showcase.csv")
                self._contact_resolve_warned = True
            return
        name_to_idx = {n: i for i, n in enumerate(shape_label)}
        try:
            with open(_CONTACT_CONFIG, "r", encoding="utf-8") as f:
                cc = json.load(f)
        except Exception as exc:
            print(f"[SimStateLogger] contact_config.json load failed: {exc}; "
                  "contact columns disabled")
            self._pair_specs = []
            return

        self._pair_specs = []
        for entry in cc.get("pairs", []):
            ia = name_to_idx.get(entry.get("a"))
            ib = name_to_idx.get(entry.get("b"))
            if ia is None or ib is None:
                print(f"[SimStateLogger] contact pair shape not found: {entry.get('name')}")
                self._pair_specs.append(None)
                continue
            self._pair_specs.append({
                "name": entry.get("name", "pair"),
                "key": frozenset((ia, ib)),
                "flip": bool(entry.get("flip_direction", False)),
            })

        agg_entries = cc.get("aggregate_groups", [])
        if agg_entries:
            entry = agg_entries[0]
            keys: set = set()
            for p in entry.get("pairs", []):
                ia = name_to_idx.get(p.get("a"))
                ib = name_to_idx.get(p.get("b"))
                if ia is not None and ib is not None:
                    keys.add(frozenset((ia, ib)))
            self._aggregate_spec = {
                "name": entry.get("name", "aggregate"),
                "pair_keys": keys,
                "origin_shape_idx": name_to_idx.get(entry.get("origin_shape")),
                "axis": tuple(float(x) for x in (entry.get("axis") or [1.0, 0.0, 0.0])),
            }
        else:
            self._aggregate_spec = None
        print(f"[SimStateLogger] contact cols resolved: pairs={len(self._pair_specs)}, "
              f"aggregate={'yes' if self._aggregate_spec else 'no'}")

    def _capture_contacts(self) -> None:
        """Per-step contact extraction (ports old showcase_logger._collect_row).

        Appends per-step values to self._buf['contact_*']. On error or no
        contact, zeros are appended so all buffers stay step-aligned with time.
        """
        # Default zeros — guarantees row alignment even when extraction fails.
        n_pairs = len(self._pair_specs or [])
        pair_pos      = np.zeros((n_pairs, 3), dtype=np.float32)
        pair_scalar   = np.zeros(n_pairs,      dtype=np.float32)
        pair_vec      = np.zeros((n_pairs, 3), dtype=np.float32)
        agg_origin    = np.zeros(3,            dtype=np.float32)
        agg_total     = np.float32(0.0)
        agg_normal    = np.zeros(3,            dtype=np.float32)

        if self._pair_specs is None or (not self._pair_specs and self._aggregate_spec is None):
            self._buf["contact_pair_pos"].append(pair_pos)
            self._buf["contact_pair_scalar"].append(pair_scalar)
            self._buf["contact_pair_vec"].append(pair_vec)
            self._buf["contact_agg_origin"].append(agg_origin)
            self._buf["contact_agg_total"].append(agg_total)
            self._buf["contact_agg_normal"].append(agg_normal)
            return

        try:
            import warp as wp
            mjw_data = getattr(self._solver, "mjw_data", None)
            mjc_geom_to_newton_shape = getattr(self._solver, "mjc_geom_to_newton_shape", None)
            if mjw_data is None or mjc_geom_to_newton_shape is None:
                raise RuntimeError("mjw_data / geom_to_shape unavailable")
            nacon_arr = getattr(mjw_data, "nacon", None)
            nacon = int(nacon_arr.numpy()[0]) if nacon_arr is not None else 0
            if nacon > 0:
                contact = mjw_data.contact
                geom_np         = wp.to_torch(contact.geom)[:nacon].cpu().numpy()
                pos_np          = wp.to_torch(contact.pos)[:nacon].cpu().numpy()
                normal_np       = wp.to_torch(contact.frame)[:nacon, 0, :].cpu().numpy()
                efc_addr_np     = wp.to_torch(contact.efc_address)[:nacon, 0].cpu().numpy()
                worldid_np      = wp.to_torch(contact.worldid)[:nacon].cpu().numpy()
                efc_force_np    = wp.to_torch(mjw_data.efc.force).cpu().numpy()
                geom_to_shape_np = wp.to_torch(mjc_geom_to_newton_shape).cpu().numpy()
                sh0 = geom_to_shape_np[worldid_np, geom_np[:, 0]]
                sh1 = geom_to_shape_np[worldid_np, geom_np[:, 1]]
                aggregate_active = False
                for i in range(nacon):
                    if sh0[i] < 0 or sh1[i] < 0:
                        continue
                    addr = int(efc_addr_np[i]); wid = int(worldid_np[i])
                    if addr < 0:
                        continue
                    scalar = float(efc_force_np[wid, addr])
                    fmag = abs(scalar)
                    fvec = (-scalar) * normal_np[i]  # reaction force on body
                    pair_key = frozenset((int(sh0[i]), int(sh1[i])))
                    for pi, ps in enumerate(self._pair_specs):
                        if ps is None or pair_key != ps["key"]:
                            continue
                        sign = -1.0 if ps["flip"] else 1.0
                        pair_pos[pi]    = pos_np[i].astype(np.float32)
                        pair_scalar[pi] = np.float32(sign * fmag)
                        pair_vec[pi]    = (sign * fvec).astype(np.float32)
                        break
                    if (self._aggregate_spec is not None
                            and pair_key in self._aggregate_spec["pair_keys"]):
                        agg_total = np.float32(float(agg_total) + fmag)
                        aggregate_active = True
                if aggregate_active and self._aggregate_spec is not None:
                    sidx = self._aggregate_spec["origin_shape_idx"]
                    if sidx is not None:
                        matches = np.where(geom_to_shape_np == sidx)
                        if matches[0].size > 0:
                            w = int(matches[0][0]); g = int(matches[1][0])
                            xpos_arr = getattr(mjw_data, "geom_xpos", None)
                            xmat_arr = getattr(mjw_data, "geom_xmat", None)
                            if xpos_arr is not None and xmat_arr is not None:
                                xpos = xpos_arr.numpy()
                                xmat = xmat_arr.numpy()
                                pos  = xpos[w, g] if xpos.ndim == 3 else xpos[g]
                                agg_origin = pos.astype(np.float32)
                                if   xmat.ndim == 4:                          rot = xmat[w, g]
                                elif xmat.ndim == 3 and xmat.shape[-1] == 9: rot = xmat[w, g].reshape(3, 3)
                                elif xmat.ndim == 3:                          rot = xmat[g]
                                elif xmat.ndim == 2 and xmat.shape[-1] == 9: rot = xmat[g].reshape(3, 3)
                                else:                                          rot = None
                                if rot is not None:
                                    ax = np.asarray(self._aggregate_spec["axis"], dtype=np.float64)
                                    nrm = np.asarray(rot, dtype=np.float64) @ ax
                                    nlen = float(np.linalg.norm(nrm))
                                    if nlen > 1e-9:
                                        agg_normal = (nrm / nlen).astype(np.float32)
        except Exception as exc:
            if not self._contact_resolve_warned:
                print(f"[SimStateLogger] contact extraction failed (logged once): {exc}")
                self._contact_resolve_warned = True

        self._buf["contact_pair_pos"].append(pair_pos)
        self._buf["contact_pair_scalar"].append(pair_scalar)
        self._buf["contact_pair_vec"].append(pair_vec)
        self._buf["contact_agg_origin"].append(agg_origin)
        self._buf["contact_agg_total"].append(agg_total)
        self._buf["contact_agg_normal"].append(agg_normal)

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

        # Joint target position from Newton control (= ArticulationAction.joint_positions
        # — what MotorStateMirror sees as q_target each step).
        tgt_out = np.zeros(len(self._torque_idx), dtype=np.float32)
        if self._control is not None:
            tgt_arr = getattr(self._control, "joint_target_pos", None)
            if tgt_arr is not None:
                try:
                    qt_raw = tgt_arr.numpy() if hasattr(tgt_arr, "numpy") else np.asarray(tgt_arr)
                    qt = (qt_raw[0] if qt_raw.ndim == 2 else qt_raw).astype(np.float32)
                    tgt_out = np.array([qt[i] if i < len(qt) else 0.0
                                        for i in self._torque_idx], dtype=np.float32)
                except Exception:
                    pass
        self._buf["joint_position_target"].append(tgt_out)

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
                applied = self._slice0(mjw_data.qfrc_applied).astype(np.float32)
                applied_out = np.array([applied[i] if i < len(applied) else 0.0
                                        for i in self._torque_idx], dtype=np.float32)
            except Exception:
                applied_out = np.zeros(n_torque, dtype=np.float32)
            try:
                grav = self._slice0(mjw_data.qfrc_gravcomp).astype(np.float32)
            except Exception:
                grav = None
        else:
            act_out = np.zeros(n_torque, dtype=np.float32)
            applied_out = np.zeros(n_torque, dtype=np.float32)
            grav = None

        self._buf["joint_torque"].append(act_out)
        self._buf["qfrc_applied"].append(applied_out)

        # qfrc_gravcomp 전체 active motor 에 대해 기록 (gravcomp magnitude 검증용)
        if grav is not None:
            grav_all = np.array([grav[i] if i < len(grav) else 0.0
                                 for i in self._torque_idx], dtype=np.float32)
        else:
            grav_all = np.zeros(n_torque, dtype=np.float32)
        self._buf["qfrc_gravcomp"].append(grav_all)

        if grav is not None and self._passive_idx:
            passive = np.array([grav[i] if i < len(grav) else 0.0
                                for i in self._passive_idx], dtype=np.float32)
        else:
            passive = np.zeros(len(self._passive_names), dtype=np.float32)
        self._buf["passive_torque"].append(passive)

        # MotorStateMirror live K_m_host / Kv_m_host / trq_m_host per active joint.
        # These are the per-step ramped motor-domain values driving the kernel.
        Km   = np.zeros(n_torque, dtype=np.float32)
        Kvm  = np.zeros(n_torque, dtype=np.float32)
        Tm   = np.zeros(n_torque, dtype=np.float32)
        mirror = getattr(self._scenario_ref, "_motor_mirror", None) if self._scenario_ref else None
        if mirror is not None:
            mi = getattr(mirror, "_motor_index", None)
            if mi is not None:
                for col, jname in enumerate(self._torque_names):
                    slot_info = mi.get(jname)
                    if slot_info is None:
                        continue
                    grp, slot = slot_info
                    Km[col]  = float(grp.K_m_host[slot])
                    Kvm[col] = float(grp.Kv_m_host[slot])
                    Tm[col]  = float(grp.trq_m_host[slot])
        self._buf["K_m_host"].append(Km)
        self._buf["Kv_m_host"].append(Kvm)
        self._buf["trq_m_host"].append(Tm)

        # Contact extraction (showcase.csv) — lazy resolve, always append even
        # when no contacts so all buffers stay aligned with self._time_buf.
        self._resolve_contact_columns()
        self._capture_contacts()

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
        tgt_cols = ["tgt_" + n.split("/")[-1] for n in self._torque_names]

        specs: list[tuple[str, str, list[str]]] = [
            ("joint_position.csv",        "joint_position",        pos_cols),
            ("joint_position_target.csv", "joint_position_target", tgt_cols),
            ("K_m_host.csv",      "K_m_host",      self._torque_names),
            ("Kv_m_host.csv",     "Kv_m_host",     self._torque_names),
            ("trq_m_host.csv",    "trq_m_host",    self._torque_names),
            ("joint_torque.csv",   "joint_torque",   self._torque_names),
            ("qfrc_applied.csv",   "qfrc_applied",   self._torque_names),
            ("qfrc_gravcomp.csv",  "qfrc_gravcomp",  self._torque_names),
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

        self._write_showcase_csv(out_dir, time_arr)
        print(f"[SimStateLogger] {n} samples saved → {out_dir}")
        return out_dir

    def _write_showcase_csv(self, out_dir: str, time_arr: np.ndarray) -> None:
        """Combined CSV consumed by allex.replay.showcase_reader.ShowcaseReader.

        Layout (in order):
            time
            pos_<joint>                                 — rad
            torque_<joint>                              — N·m, qfrc_applied + qfrc_gravcomp
            contact_pos_<pair>_{x,y,z}                  — m, world frame
            aggregate_origin_{x,y,z}                    — m, world frame (if any)
            force_<pair>                                — signed scalar N
            force_aggregate_<group>                     — sum |scalar| N (if any)
            force_vec_<pair>_{x,y,z}                    — N, world frame
            normal_<group>_{x,y,z}                      — unit vector (if any)

        Torque column = qfrc_applied + qfrc_gravcomp (matches real-robot
        joint_torque topic semantics: motor PD output + gravity feed-forward,
        produced by MotorStateMirror kernel after ke=kd=0 switch).
        """
        pos_data     = np.asarray(self._buf["joint_position"], dtype=np.float64)
        applied_data = np.asarray(self._buf["qfrc_applied"],   dtype=np.float64)
        grav_data    = np.asarray(self._buf["qfrc_gravcomp"],  dtype=np.float64)
        if pos_data.size == 0:
            return
        torque_data = applied_data + grav_data   # showcase torque convention

        pair_pos_arr   = np.asarray(self._buf["contact_pair_pos"],    dtype=np.float64)  # (n, p, 3)
        pair_scalar    = np.asarray(self._buf["contact_pair_scalar"], dtype=np.float64)  # (n, p)
        pair_vec_arr   = np.asarray(self._buf["contact_pair_vec"],    dtype=np.float64)  # (n, p, 3)
        agg_origin_arr = np.asarray(self._buf["contact_agg_origin"],  dtype=np.float64)  # (n, 3)
        agg_total_arr  = np.asarray(self._buf["contact_agg_total"],   dtype=np.float64)  # (n,)
        agg_normal_arr = np.asarray(self._buf["contact_agg_normal"],  dtype=np.float64)  # (n, 3)

        has_pairs = pair_pos_arr.ndim == 3 and pair_pos_arr.shape[1] > 0
        has_agg   = self._aggregate_spec is not None

        # ---- header ----
        cols: list[str] = ["time"]
        cols += [f"pos_{n}"    for n in self._torque_names]
        cols += [f"torque_{n}" for n in self._torque_names]
        pair_names: list[str] = []
        if has_pairs and self._pair_specs:
            pair_names = [
                (ps["name"] if ps is not None else f"pair_{i}_unresolved")
                for i, ps in enumerate(self._pair_specs)
            ]
            for nm in pair_names:
                cols += [f"contact_pos_{nm}_{ax}" for ax in ("x", "y", "z")]
        if has_agg:
            cols += [f"aggregate_origin_{ax}" for ax in ("x", "y", "z")]
        for nm in pair_names:
            cols.append(f"force_{nm}")
        if has_agg:
            cols.append(f"force_aggregate_{self._aggregate_spec['name']}")
        for nm in pair_names:
            cols += [f"force_vec_{nm}_{ax}" for ax in ("x", "y", "z")]
        if has_agg:
            cols += [f"normal_{self._aggregate_spec['name']}_{ax}" for ax in ("x", "y", "z")]

        fpath = os.path.join(out_dir, "showcase.csv")
        with open(fpath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            n_steps = len(time_arr)
            for k in range(n_steps):
                row: list = [f"{time_arr[k]:.4f}"]
                row += [f"{v:.6f}" for v in pos_data[k]]
                row += [f"{v:.6f}" for v in torque_data[k]]
                if has_pairs:
                    for p in range(pair_pos_arr.shape[1]):
                        row += [f"{v:.6f}" for v in pair_pos_arr[k, p]]
                if has_agg:
                    row += [f"{v:.6f}" for v in agg_origin_arr[k]]
                if has_pairs:
                    row += [f"{v:.6f}" for v in pair_scalar[k]]
                if has_agg:
                    row.append(f"{float(agg_total_arr[k]):.6f}")
                if has_pairs:
                    for p in range(pair_vec_arr.shape[1]):
                        row += [f"{v:.6f}" for v in pair_vec_arr[k, p]]
                if has_agg:
                    row += [f"{v:.6f}" for v in agg_normal_arr[k]]
                w.writerow(row)
