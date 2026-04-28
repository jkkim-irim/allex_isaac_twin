"""Per-physics-step CSV logger for trajectory playback showcase data.

Recording starts when TrajStudio's Run is pressed and auto-stops when the
trajectory player goes inactive (or stop_recording() is called explicitly).

CSV layout (column groups, in order):
    time
    pos_<active_joint_name>      x 48      (joint_config.json active_joints order)
    torque_<active_joint_name>   x 48
    contact_pos_<pair_i>_{x,y,z} x 2 pairs (zero-padded if no contact)
    aggregate_origin_{x,y,z}             (zero if aggregate inactive)
    force_<pair_i>               x 2 pairs (signed if flip_direction=true)
    force_aggregate_<group>              (sum of |scalar| across group's pairs)
    force_vec_<pair_i>_{x,y,z}   x 2 pairs world frame
    normal_<group>_{x,y,z}               (origin shape's local axis in world)
"""
from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path


_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_SHOWCASE_DIR = _DATA_DIR / "showcase"
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_CONTACT_CONFIG = _CONFIG_DIR / "contact_config.json"
_JOINT_CONFIG = _CONFIG_DIR / "joint_config.json"
_LOGGER_CONFIG = _CONFIG_DIR / "showcase_logger_config.json"


def is_group_logged(group_name: str) -> bool:
    """Return True if `group_name` is in `showcase_logger_config.json::logged_groups`.

    Allows the Traj Studio "Run" handler to decide whether to start CSV
    recording for a given trajectory group. Missing config / missing key /
    non-list value → returns False (record nothing).
    """
    if not group_name:
        return False
    try:
        with open(_LOGGER_CONFIG, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        return False
    except Exception as exc:
        print(f"[ALLEX][Showcase] failed to read {_LOGGER_CONFIG.name}: {exc}")
        return False
    groups = cfg.get("logged_groups")
    if not isinstance(groups, list):
        return False
    return group_name in groups

# Pose-aligned start: defer CSV writing until the live articulation pose
# matches the first frame of this rosbag (i.e., crop both the 1.5s ramp-in
# and the trajectory transit so sim t=0 == real t=0). Set to None or point
# at a missing file to record from player.start() as before.
_ALIGN_ROSBAG = (
    _DATA_DIR / "rosbag2_2026_04_24-04_14_41_demo1_dynamic"
              / "rosbag2_2026_04_24-04_14_41_0.mcap"
)
_ALIGN_RMS_THRESHOLD_RAD = 0.05   # ~3° per-joint RMS — "same pose"
_ALIGN_TIMEOUT_S = 30.0           # safety net: after this, log anyway


# Joint → (rosbag topic, data index). Mirrors `_plot_torque_compare.py::ALIGN_JOINTS`.
# Hands are excluded — finger angles are noisy and contribute little to large-pose match.
def _build_align_topic_map() -> dict[str, tuple[str, int]]:
    out: dict[str, tuple[str, int]] = {}
    for side in ("R", "L"):
        topic = f"/robot_outbound_data/Arm_{side}_theOne/joint_positions_deg"
        for i, name in enumerate([
            f"{side}_Shoulder_Pitch_Joint", f"{side}_Shoulder_Roll_Joint",
            f"{side}_Shoulder_Yaw_Joint",   f"{side}_Elbow_Joint",
            f"{side}_Wrist_Yaw_Joint",      f"{side}_Wrist_Roll_Joint",
            f"{side}_Wrist_Pitch_Joint",
        ]):
            out[name] = (topic, i)
    out["Waist_Yaw_Joint"]   = ("/robot_outbound_data/theOne_waist/joint_positions_deg", 0)
    out["Neck_Pitch_Joint"]  = ("/robot_outbound_data/theOne_neck/joint_positions_deg", 0)
    out["Neck_Yaw_Joint"]    = ("/robot_outbound_data/theOne_neck/joint_positions_deg", 1)
    return out


def _load_rosbag_first_pose(bag_path: Path) -> dict[str, float]:
    """Read the first message of each align topic and return joint→radians."""
    import numpy as np
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    align_map = _build_align_topic_map()
    needed = set(t for (t, _) in align_map.values())
    first_msg: dict[str, list[float]] = {}
    with bag_path.open("rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, channel, _msg, ros_msg in reader.iter_decoded_messages(topics=list(needed)):
            if channel.topic in first_msg:
                continue
            data = list(getattr(ros_msg, "data", []) or [])
            if not data:
                continue
            first_msg[channel.topic] = data
            if len(first_msg) == len(needed):
                break

    pose: dict[str, float] = {}
    for jname, (topic, idx) in align_map.items():
        row = first_msg.get(topic)
        if row is not None and idx < len(row):
            pose[jname] = float(np.deg2rad(row[idx]))
    return pose


class ShowcaseDataLogger:
    """CSV logger for trajectory playback runs."""

    def __init__(self) -> None:
        self._recording: bool = False
        self._articulation = None
        self._player = None
        self._physics_dt: float = 0.0
        self._step_idx: int = 0
        self._csv_file = None
        self._csv_writer = None
        self._output_path: Path | None = None
        # Resolved column descriptors at start time.
        # Position columns: all active joints (incl. passive followers).
        # Torque columns: only actuated joints (drive_gains.stiffness > 0).
        self._pos_dof_indices: list[int] = []
        self._pos_dof_names: list[str] = []
        self._torque_dof_indices: list[int] = []
        self._torque_dof_names: list[str] = []
        # Each pair: {"name", "key": frozenset, "flip": bool} or None on resolve fail.
        self._pair_specs: list = []
        # Single aggregate group (first in JSON). None if absent.
        self._aggregate_spec: dict | None = None
        # Pose-aligned start: when set, defer CSV row writes until the live
        # pose matches the rosbag's first frame within `_ALIGN_RMS_THRESHOLD_RAD`.
        self._align_target_rad: dict[str, float] | None = None
        self._aligned: bool = True
        self._align_wait_steps: int = 0
        self._align_timeout_steps: int = 0
        # name → dof_idx for the joints used in the pose-match metric;
        # populated lazily on the first aligned check.
        self._align_dof_indices: dict[str, int] | None = None

    # ------------------------------------------------------------------
    # Public API (called from TrajStudio)
    # ------------------------------------------------------------------
    def start_recording(self, articulation, player, physics_dt: float):
        if self._recording:
            self.stop_recording()
        try:
            self._resolve_columns(articulation)
            _SHOWCASE_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._output_path = _SHOWCASE_DIR / f"showcase_sim_data_{ts}.csv"
            self._csv_file = open(self._output_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(self._build_header())
            self._articulation = articulation
            self._player = player
            self._physics_dt = float(physics_dt)
            self._step_idx = 0
            self._recording = True

            # Pose-aligned start: try to load rosbag first frame. If that
            # succeeds, defer CSV writes until live pose is close enough.
            self._align_target_rad = None
            self._aligned = True
            self._align_wait_steps = 0
            self._align_dof_indices = None
            self._align_timeout_steps = max(
                1, int(round(_ALIGN_TIMEOUT_S / max(self._physics_dt, 1e-6)))
            )
            if _ALIGN_ROSBAG is not None and _ALIGN_ROSBAG.exists():
                try:
                    self._align_target_rad = _load_rosbag_first_pose(_ALIGN_ROSBAG)
                    if self._align_target_rad:
                        self._aligned = False
                        print(
                            f"[ALLEX][Showcase] CSV start deferred until pose "
                            f"match with {_ALIGN_ROSBAG.name} "
                            f"({len(self._align_target_rad)} joints, "
                            f"RMS<{_ALIGN_RMS_THRESHOLD_RAD:.3f} rad, "
                            f"timeout {_ALIGN_TIMEOUT_S:.0f}s)"
                        )
                except Exception as exc:
                    print(
                        f"[ALLEX][Showcase] failed to read align rosbag: {exc}; "
                        f"recording from player.start()"
                    )
                    self._align_target_rad = None
                    self._aligned = True

            print(f"[ALLEX][Showcase] recording started: {self._output_path}")
            return self._output_path
        except Exception as exc:
            print(f"[ALLEX][Showcase] start failed: {exc}")
            self._cleanup_files()
            return None

    def stop_recording(self) -> None:
        if not self._recording:
            return
        self._recording = False
        n = self._step_idx
        self._cleanup_files()
        print(f"[ALLEX][Showcase] recording stopped after {n} rows: {self._output_path}")
        self._articulation = None
        self._player = None

    def cleanup(self) -> None:
        self.stop_recording()

    # ------------------------------------------------------------------
    # Per-physics-step entry
    # ------------------------------------------------------------------
    def on_physics_step(self, step_dt: float) -> None:
        if not self._recording or self._csv_writer is None:
            return
        # Stop on natural completion (`is_finished`) too, not just external stop.
        # The TrajectoryPlayer flips `_finished=True` at trajectory end but keeps
        # `_active=True` until stop() is called externally — without checking
        # _finished we'd loop forever on the last row and never close the file.
        if self._player is not None:
            if self._player.is_finished() or not self._player.is_active():
                self.stop_recording()
                return

        # Pose-aligned start: hold off until the live pose matches the rosbag's
        # first frame (or we time out). Once activated, _step_idx restarts from
        # 0 so the CSV's `time` column is rosbag-aligned with t=0.
        if not self._aligned:
            self._align_wait_steps += 1
            rms = self._align_pose_rms()
            if rms is not None and rms <= _ALIGN_RMS_THRESHOLD_RAD:
                wait_s = self._align_wait_steps * self._physics_dt
                print(
                    f"[ALLEX][Showcase] pose-aligned after {wait_s:.3f}s "
                    f"({self._align_wait_steps} steps, RMS={rms:.4f} rad); "
                    f"CSV t=0 begins"
                )
                self._aligned = True
                self._step_idx = 0
            elif self._align_wait_steps >= self._align_timeout_steps:
                print(
                    f"[ALLEX][Showcase] alignment timeout "
                    f"({_ALIGN_TIMEOUT_S:.0f}s, last RMS="
                    f"{rms if rms is not None else float('nan'):.4f} rad); "
                    f"recording from current pose"
                )
                self._aligned = True
                self._step_idx = 0
            else:
                return

        try:
            row = self._collect_row()
            if row is not None:
                # Round all float cells to 6 decimals — strips float64 noise
                # tail (~15-17 digits) while keeping sensor/sim precision well
                # above any meaningful signal floor.
                self._csv_writer.writerow([
                    round(v, 6) if isinstance(v, float) else v for v in row
                ])
            self._step_idx += 1
            # Periodic flush so the file is visible/usable mid-run.
            if self._step_idx % 200 == 0 and self._csv_file is not None:
                self._csv_file.flush()
        except Exception as exc:
            print(f"[ALLEX][Showcase] row collect failed at step {self._step_idx}: {exc}")
            self.stop_recording()

    # ------------------------------------------------------------------
    # Pose alignment
    # ------------------------------------------------------------------
    def _align_pose_rms(self) -> float | None:
        """Return RMS distance (rad) between live pose and rosbag first frame.

        Returns None when the pose cannot be read or there are no overlapping
        joints. The first call resolves and caches `name → dof_idx`.
        """
        if not self._align_target_rad or self._articulation is None:
            return None
        try:
            if self._align_dof_indices is None:
                names = list(getattr(self._articulation, "dof_names", []) or [])
                name_to_dof = {n: i for i, n in enumerate(names)}
                self._align_dof_indices = {
                    j: name_to_dof[j]
                    for j in self._align_target_rad
                    if j in name_to_dof
                }
                if not self._align_dof_indices:
                    print("[ALLEX][Showcase] pose-align: no overlapping joints "
                          "between rosbag pose and articulation; disabling")
                    self._align_target_rad = None
                    self._aligned = True
                    return None
            qpos = self._to_numpy_1d(self._articulation.get_joint_positions())
            if qpos is None:
                return None
            sq = 0.0
            n = 0
            for j, idx in self._align_dof_indices.items():
                if idx >= len(qpos):
                    continue
                d = float(qpos[idx]) - self._align_target_rad[j]
                sq += d * d
                n += 1
            if n == 0:
                return None
            return (sq / n) ** 0.5
        except Exception as exc:
            print(f"[ALLEX][Showcase] pose-align RMS failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _cleanup_files(self) -> None:
        if self._csv_file is not None:
            try:
                self._csv_file.close()
            except Exception:
                pass
        self._csv_file = None
        self._csv_writer = None

    def _resolve_columns(self, articulation) -> None:
        # Position columns: every entry in joint_config.json::active_joints.
        # Torque columns: same set filtered to joints with non-zero authored
        # drive stiffness (passive followers like *_DIP / Waist_Upper_Pitch /
        # Waist_Pitch_Dummy have kp=0 and never receive actuator torque, so
        # logging their torque is meaningless).
        with open(_JOINT_CONFIG) as f:
            jc = json.load(f)
        joint_names_map = jc.get("joint_names", {})
        active_indices_str = jc.get("active_joints", [])
        drive_gains = jc.get("drive_gains", {}) or {}
        active_joint_names = [
            joint_names_map[i] for i in active_indices_str if i in joint_names_map
        ]
        dof_names = list(getattr(articulation, "dof_names", []) or [])
        name_to_dof = {n: i for i, n in enumerate(dof_names)}

        self._pos_dof_indices = []
        self._pos_dof_names = []
        self._torque_dof_indices = []
        self._torque_dof_names = []
        skipped_passive: list[str] = []
        for nm in active_joint_names:
            idx = name_to_dof.get(nm)
            if idx is None:
                print(f"[ALLEX][Showcase] active joint '{nm}' not in articulation.dof_names")
                continue
            self._pos_dof_indices.append(idx)
            self._pos_dof_names.append(nm)
            stiffness = float((drive_gains.get(nm) or {}).get("stiffness", 0.0))
            if stiffness > 0.0:
                self._torque_dof_indices.append(idx)
                self._torque_dof_names.append(nm)
            else:
                skipped_passive.append(nm)
        print(
            f"[ALLEX][Showcase] columns: pos={len(self._pos_dof_names)}, "
            f"torque={len(self._torque_dof_names)} "
            f"(skipped {len(skipped_passive)} passive: {skipped_passive})"
        )

        # Contact config — need shape_label to resolve names.
        from isaacsim.physics.newton import acquire_stage
        stage = acquire_stage()
        model = getattr(stage, "model", None) if stage is not None else None
        shape_label = list(getattr(model, "shape_label", None) or [])
        if not shape_label:
            raise RuntimeError("Newton model.shape_label not ready")
        name_to_idx = {n: i for i, n in enumerate(shape_label)}

        with open(_CONTACT_CONFIG) as f:
            cc = json.load(f)

        self._pair_specs = []
        for entry in cc.get("pairs", []):
            ia = name_to_idx.get(entry.get("a"))
            ib = name_to_idx.get(entry.get("b"))
            if ia is None or ib is None:
                print(f"[ALLEX][Showcase] pair shape not found: {entry.get('name')}")
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
            origin_shape_idx = name_to_idx.get(entry.get("origin_shape"))
            self._aggregate_spec = {
                "name": entry.get("name", "aggregate"),
                "pair_keys": keys,
                "origin_shape_idx": origin_shape_idx,
                "axis": tuple(float(x) for x in (entry.get("axis") or [1.0, 0.0, 0.0])),
            }
        else:
            self._aggregate_spec = None

    def _build_header(self) -> list[str]:
        cols: list[str] = ["time"]
        cols += [f"pos_{n}" for n in self._pos_dof_names]
        cols += [f"torque_{n}" for n in self._torque_dof_names]
        for i, ps in enumerate(self._pair_specs):
            base = ps["name"] if ps is not None else f"pair_{i}_unresolved"
            cols += [f"contact_pos_{base}_{ax}" for ax in ("x", "y", "z")]
        if self._aggregate_spec is not None:
            cols += [f"aggregate_origin_{ax}" for ax in ("x", "y", "z")]
        for i, ps in enumerate(self._pair_specs):
            base = ps["name"] if ps is not None else f"pair_{i}_unresolved"
            cols += [f"force_{base}"]
        if self._aggregate_spec is not None:
            cols += [f"force_aggregate_{self._aggregate_spec['name']}"]
        for i, ps in enumerate(self._pair_specs):
            base = ps["name"] if ps is not None else f"pair_{i}_unresolved"
            cols += [f"force_vec_{base}_{ax}" for ax in ("x", "y", "z")]
        if self._aggregate_spec is not None:
            cols += [f"normal_{self._aggregate_spec['name']}_{ax}" for ax in ("x", "y", "z")]
        return cols

    def _read_torques(self):
        """Read total per-DOF feedforward torque = PD + gravity compensation.

        Bypasses `articulation.get_*_joint_efforts()` (Newton's
        `get_dof_projected_joint_forces` is not implemented and warns each
        call). With `mjc:actuatorgravcomp = false` (passive routing) the PD
        controller output sits in `qfrc_actuator` and the feedforward gravity
        compensation sits in `qfrc_gravcomp` — summing them gives the joint's
        total commanded torque, the apples-to-apples counterpart of the real
        robot's `joint_torque` topic (which already includes the controller's
        own gravity feed-forward).
        """
        import numpy as np
        try:
            from isaacsim.physics.newton import acquire_stage
            stage = acquire_stage()
            solver = getattr(stage, "solver", None) if stage is not None else None
            mjw_data = getattr(solver, "mjw_data", None) if solver is not None else None
            if mjw_data is None:
                return None

            def _slice0(arr):
                np_arr = arr.numpy()
                return np_arr[0] if np_arr.ndim == 2 else np_arr

            # PD output (always populated by MuJoCo Warp once actuators exist).
            pd = None
            for attr in ("qfrc_actuator", "qfrc_smooth", "qfrc_applied"):
                a = getattr(mjw_data, attr, None)
                if a is None:
                    continue
                try:
                    pd = _slice0(a)
                    break
                except Exception:
                    continue
            if pd is None:
                return None

            # Gravity compensation — present only when `m.ngravcomp > 0`.
            # If absent (no body has nonzero `mjc:gravcomp`), the kernel is
            # short-circuited and the array doesn't get populated; treat it
            # as zero rather than failing the whole read.
            grav = None
            grav_arr = getattr(mjw_data, "qfrc_gravcomp", None)
            if grav_arr is not None:
                try:
                    grav = _slice0(grav_arr)
                except Exception:
                    grav = None
            if grav is None or grav.shape != pd.shape:
                return pd
            return pd + grav
        except Exception:
            pass
        return None

    @staticmethod
    def _to_numpy_1d(x):
        """Coerce torch.Tensor / warp.array / list to numpy 1-D, handling CUDA tensors."""
        import numpy as np
        if x is None:
            return None
        # torch.Tensor (cuda or cpu) — has .cpu() and .numpy()
        if hasattr(x, "cpu") and callable(x.cpu):
            try:
                x = x.cpu().numpy()
            except Exception:
                pass
        # warp.array / similar — has .numpy()
        elif hasattr(x, "numpy") and callable(x.numpy):
            try:
                x = x.numpy()
            except Exception:
                pass
        arr = np.asarray(x)
        if arr.ndim > 1:
            arr = arr.reshape(-1) if arr.shape[0] == 1 else arr[0]
        return arr

    def _collect_row(self):
        import numpy as np
        from isaacsim.physics.newton import acquire_stage

        row: list = []
        # Round to 6 decimals so step_idx*dt floating-point accumulation noise
        # doesn't leak into the CSV (e.g., 0.175000000000002 → 0.175).
        row.append(round(self._step_idx * self._physics_dt, 6))

        # Joint positions (all configured active joints, incl. passives).
        try:
            qpos = self._to_numpy_1d(self._articulation.get_joint_positions())
        except Exception:
            qpos = None
        if qpos is None:
            row.extend([0.0] * len(self._pos_dof_indices))
        else:
            row.extend(float(qpos[di]) if di < len(qpos) else 0.0
                       for di in self._pos_dof_indices)

        # Joint torques (only actuated joints — passives have no actuator).
        torques = self._to_numpy_1d(self._read_torques())
        if torques is None:
            row.extend([0.0] * len(self._torque_dof_indices))
        else:
            row.extend(float(torques[di]) if di < len(torques) else 0.0
                       for di in self._torque_dof_indices)

        # Contact extraction (Newton mjw_data)
        n_pairs = len(self._pair_specs)
        pair_pos = [[0.0, 0.0, 0.0] for _ in range(n_pairs)]
        pair_force_scalar = [0.0] * n_pairs
        pair_force_vec = [[0.0, 0.0, 0.0] for _ in range(n_pairs)]
        agg_origin = [0.0, 0.0, 0.0]
        agg_normal = [0.0, 0.0, 0.0]
        agg_total_force = 0.0
        aggregate_active = False

        try:
            stage = acquire_stage()
            solver = getattr(stage, "solver", None) if stage is not None else None
            mjw_data = getattr(solver, "mjw_data", None) if solver is not None else None
            mjc_geom_to_newton_shape = getattr(solver, "mjc_geom_to_newton_shape", None) if solver is not None else None
            if mjw_data is not None and mjc_geom_to_newton_shape is not None:
                import warp as wp
                nacon_arr = getattr(mjw_data, "nacon", None)
                nacon = int(nacon_arr.numpy()[0]) if nacon_arr is not None else 0
                if nacon > 0:
                    contact = mjw_data.contact
                    geom_np = wp.to_torch(contact.geom)[:nacon].cpu().numpy()
                    pos_np = wp.to_torch(contact.pos)[:nacon].cpu().numpy()
                    normal_np = wp.to_torch(contact.frame)[:nacon, 0, :].cpu().numpy()
                    efc_addr_np = wp.to_torch(contact.efc_address)[:nacon, 0].cpu().numpy()
                    worldid_np = wp.to_torch(contact.worldid)[:nacon].cpu().numpy()
                    efc_force_np = wp.to_torch(mjw_data.efc.force).cpu().numpy()
                    geom_to_shape_np = wp.to_torch(mjc_geom_to_newton_shape).cpu().numpy()
                    sh0 = geom_to_shape_np[worldid_np, geom_np[:, 0]]
                    sh1 = geom_to_shape_np[worldid_np, geom_np[:, 1]]
                    for i in range(nacon):
                        if sh0[i] < 0 or sh1[i] < 0:
                            continue
                        addr = int(efc_addr_np[i]); wid = int(worldid_np[i])
                        if addr < 0:
                            continue
                        scalar = float(efc_force_np[wid, addr])
                        fmag = abs(scalar)
                        fvec = (-scalar) * normal_np[i]   # reaction force
                        pair_key = frozenset((int(sh0[i]), int(sh1[i])))
                        for pi, ps in enumerate(self._pair_specs):
                            if ps is None or pair_key != ps["key"]:
                                continue
                            pair_pos[pi] = [float(pos_np[i, 0]), float(pos_np[i, 1]), float(pos_np[i, 2])]
                            sign = -1.0 if ps["flip"] else 1.0
                            pair_force_scalar[pi] = sign * fmag
                            pair_force_vec[pi] = [float(sign * fvec[0]),
                                                  float(sign * fvec[1]),
                                                  float(sign * fvec[2])]
                            break
                        if self._aggregate_spec is not None and pair_key in self._aggregate_spec["pair_keys"]:
                            agg_total_force += fmag
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
                                    pos = xpos[w, g] if xpos.ndim == 3 else xpos[g]
                                    agg_origin = [float(pos[0]), float(pos[1]), float(pos[2])]
                                    if xmat.ndim == 4:
                                        rot = xmat[w, g]
                                    elif xmat.ndim == 3 and xmat.shape[-1] == 9:
                                        rot = xmat[w, g].reshape(3, 3)
                                    elif xmat.ndim == 3:
                                        rot = xmat[g]
                                    elif xmat.ndim == 2 and xmat.shape[-1] == 9:
                                        rot = xmat[g].reshape(3, 3)
                                    else:
                                        rot = None
                                    if rot is not None:
                                        ax = np.asarray(self._aggregate_spec["axis"], dtype=np.float64)
                                        nrm = np.asarray(rot, dtype=np.float64) @ ax
                                        nlen = float(np.linalg.norm(nrm))
                                        if nlen > 1e-9:
                                            agg_normal = [float(nrm[0] / nlen),
                                                          float(nrm[1] / nlen),
                                                          float(nrm[2] / nlen)]
        except Exception as exc:
            if self._step_idx == 0:
                print(f"[ALLEX][Showcase] contact extraction failed (logged once): {exc}")

        # Append in JSON-doc order:
        for p in pair_pos:
            row.extend(p)
        if self._aggregate_spec is not None:
            row.extend(agg_origin)
        row.extend(pair_force_scalar)
        if self._aggregate_spec is not None:
            row.append(agg_total_force)
        for v in pair_force_vec:
            row.extend(v)
        if self._aggregate_spec is not None:
            row.extend(agg_normal)
        return row
