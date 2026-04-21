"""
ForceTorqueVisualizer — Real/Sim 토크 링 + 힘 화살표 겹침 시각화

- Torque 링 (40개): `torque_viz.usd` 를 reference 하여
  /ALLEX/<link>/torque_ring_real, /ALLEX/<link>/torque_ring_sim Xform 두 개 생성.
  매 step sim 은 `articulation.get_measured_joint_efforts()`, real 은
  `joint_controller.get_torque_snapshot()` 값을 사용해 xformOp:scale 로 크기 표현.
  (PhysxMimicJointAPI 는 position 만 coupling 하고 DOF 는 articulation 에 남으므로
   slave 도 `get_measured_joint_efforts()` 결과에 포함된다 → master/slave 구분 없이 40 링 통일.)
- Force 화살표 (12개): 기존 `<link>/visuals/force_viz` 를 real 역할로 그대로 사용하고,
  동일 부모 (`<link>/visuals/`) 아래에 `force_viz_sim` 신규 prim 을 추가.
  sim 은 `articulation.get_measured_joint_forces()` 의 해당 link incoming joint row
  의 선형 힘 크기를 사용. real 은 현재 데이터 없음 → 0.
- Visibility 는 set_mode("off"/"real"/"sim"/"both") + set_torque_mode("off"/"real"/"sim"/"both") 로 제어.

pxr 은 Isaac Sim 런타임에만 존재하므로 import 는 메서드 내부에서 lazy 로.
"""

from __future__ import annotations

import math
import os
import logging

import carb

from ..config.viz_config import (
    REAL_COLOR, SIM_COLOR,
    TORQUE_GAIN, TORQUE_MIN_SCALE, TORQUE_MAX_SCALE,
    FORCE_GAIN, FORCE_MIN_SCALE, FORCE_MAX_SCALE,
    FORCE_VIZ_REAL_NAME, FORCE_VIZ_SIM_NAME,
    TORQUE_RING_REAL_NAME, TORQUE_RING_SIM_NAME,
    TORQUE_VIZ_USD_PATH, FORCE_VIZ_USD_PATH,
    HAND_JOINT_TORQUE_RING_MAP, HAND_JOINT_TORQUE_RING_TUPLES,
    FORCE_VIZ_PARENT_LINKS,
    FORCE_VIZ_VISUAL_SUBPATH,
    FORCE_VEC_SHAFT_NAME, FORCE_VEC_HEAD_NAME,
    FORCE_VEC_SHAFT_BASE_LENGTH, FORCE_VEC_HEAD_BASE_TRANSLATE_Z,
    FORCE_VEC_LENGTH_AXIS,
)


logger = logging.getLogger("allex.viz")


# scale dead-band: 직전 적용 scale 과 차이가 이보다 작으면 Set skip
_SCALE_DEADBAND = 1e-3


def _clip(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def Usd_iter_descendants(prim):
    """prim 자신 제외한 모든 자손을 yield. Usd.PrimRange 대안 (pxr import 독립)."""
    stack = list(prim.GetChildren())
    while stack:
        p = stack.pop()
        yield p
        stack.extend(p.GetChildren())


class ForceTorqueVisualizer:
    """Real/Sim 겹침 시각화 매니저.

    articulation      : isaacsim.core.prims.SingleArticulation 인스턴스
    joint_controller  : ALLEXJointController 인스턴스 (real torque snapshot 제공)
    stage             : pxr.Usd.Stage (없으면 런타임에 조회)
    """

    def __init__(self, stage=None, articulation=None, joint_controller=None):
        """Prim 생성은 stage 만 있으면 바로 수행. articulation/joint_controller 는
        나중에 `attach_articulation()` 으로 지연 주입할 수 있다.

        구 인터페이스 `ForceTorqueVisualizer(articulation, joint_controller, stage=)` 도
        허용하기 위해 첫 번째 positional 이 stage 가 아닌 articulation-like 객체일
        경우를 감지하여 swap 한다 (best-effort — dof_names 속성 유무로 판별).
        """
        # Backward-compat: 예전 시그니처 호출을 자동 감지해서 처리
        if stage is not None and (
            hasattr(stage, "dof_names") or hasattr(stage, "_articulation_view")
        ):
            # 첫번째 인자가 articulation 처럼 생겼다 → 옛 시그니처
            legacy_articulation = stage
            legacy_joint_controller = articulation  # 두번째 인자
            stage = joint_controller  # 세번째 인자
            articulation = legacy_articulation
            joint_controller = legacy_joint_controller

        self._articulation = articulation
        self._joint_controller = joint_controller
        self._stage = stage

        # 상태 — 기본 OFF. UI 토글을 눌러야 표시된다.
        self._mode = "off"          # off / real / sim / both  (force arrow)
        self._torque_mode = "off"   # off / real / sim / both  (torque ring)

        # 생성한 prim 캐시
        self._torque_ring_prims: dict = {}   # dof_abbr -> {"real": prim, "sim": prim}
        self._force_viz_prims: dict = {}     # link_path -> {"real": prim, "sim": prim}

        # force 화살표 shaft/head ops 캐시 — 동적 길이 업데이트용
        # prim_path(=shaft wrapper prim path) -> {"shaft_scale_op", "head_translate_op",
        #                                         "last_scale"}
        self._force_length_cache: dict = {}

        # 우리가 직접 생성(= stage.RemovePrim 으로 지워도 되는)한 prim path 집합
        self._created_prim_paths: set = set()

        # dof_abbr → articulation dof idx (lazy build)
        self._abbr_to_dof_idx: dict = {}

        # link_path → measured_joint_forces row (= joint_idx + 1) LUT
        self._link_to_force_row: dict = {}

        # prim path → UsdGeom.XformOp(scale) 캐시 (B5)
        self._scale_op_cache: dict = {}

        # prim path → 직전에 Set 한 scale 값 (B5 dead-band)
        self._last_scale_cache: dict = {}

        self._prims_initialized = False

        # stage 가 바로 사용 가능하면 prim 을 즉시 생성한다 (articulation 불필요).
        # 실패해도 조용히 넘어가고 ensure_initialized/update 가 재시도한다.
        try:
            if self._get_stage() is not None:
                self._ensure_prims()
                self._prims_initialized = True
        except Exception as e:
            logger.warning(f"[viz] early _ensure_prims failed: {e}")

    def attach_articulation(self, articulation, joint_controller=None):
        """articulation/joint_controller 를 지연 주입. dof 매핑 LUT 를 재구축한다.

        prim 은 이미 __init__ 에서 생성되어 있을 것이므로 건드리지 않는다. 재호출
        (reset 등) 도 안전하다.
        """
        self._articulation = articulation
        if joint_controller is not None:
            self._joint_controller = joint_controller
        # prim 이 아직 안 만들어졌다면 지금이라도 생성 시도
        if not self._prims_initialized:
            try:
                self._ensure_prims()
                self._prims_initialized = True
            except Exception as e:
                logger.warning(f"[viz] _ensure_prims (attach) failed: {e}")
        # dof/force row LUT 재구축
        try:
            self._build_abbr_to_dof_idx()
            self._build_link_to_force_row()
        except Exception as e:
            logger.warning(f"[viz] attach_articulation LUT build failed: {e}")

    def detach_articulation(self):
        """articulation 참조만 해제. prim 은 그대로 유지."""
        self._articulation = None
        self._joint_controller = None
        self._abbr_to_dof_idx.clear()
        self._link_to_force_row.clear()

    # ========================================
    # Public API
    # ========================================
    def ensure_initialized(self):
        """Prim 생성 (+ articulation 이 있으면 dof 매핑까지).

        articulation 없이도 prim 은 만들어 둬야 한다(Play 전에도 표시 가능하도록).
        articulation 이 이후 `attach_articulation` 으로 들어오면 LUT 를 재구축한다.
        """
        if not self._prims_initialized:
            try:
                self._ensure_prims()
                self._prims_initialized = True
            except Exception as e:
                logger.warning(f"[viz] ensure_initialized _ensure_prims failed: {e}")
                return
        # articulation 이 이미 붙어 있으면 LUT 도 빌드 (없으면 attach 시점에 빌드)
        if self._articulation is not None:
            try:
                self._build_abbr_to_dof_idx()
                self._build_link_to_force_row()
            except Exception as e:
                logger.warning(f"[viz] ensure_initialized LUT build failed: {e}")

    def update(self, step: float = 0.0):
        """매 physics step 호출."""
        if self._mode == "off" and self._torque_mode == "off":
            return

        if not self._prims_initialized:
            self.ensure_initialized()
            if not self._prims_initialized:
                return

        # articulation 이 없으면 scale 업데이트를 건너뛴다 (prim 과 visibility 는 유지).
        if self._articulation is None or self._joint_controller is None:
            return

        # --- sim torque ---
        sim_torques = None
        try:
            if self._articulation is not None:
                sim_torques = self._articulation.get_measured_joint_efforts()
        except Exception as e:
            logger.debug(f"[viz] get_measured_joint_efforts failed: {e}")
            sim_torques = None

        # --- real torque (dict by abbr) ---
        real_snapshot = {}
        if self._joint_controller is not None:
            try:
                real_snapshot = self._joint_controller.get_torque_snapshot() or {}
            except Exception as e:
                logger.debug(f"[viz] torque snapshot failed: {e}")

        # --- sim joint forces (N+1, 6) ---
        sim_forces = None
        try:
            if self._articulation is not None and hasattr(self._articulation, "get_measured_joint_forces"):
                sim_forces = self._articulation.get_measured_joint_forces()
        except Exception as e:
            logger.debug(f"[viz] get_measured_joint_forces failed: {e}")
            sim_forces = None

        # --- torque rings (평탄화된 튜플 순회 — N3) ---
        update_sim_torque = self._torque_mode in ("sim", "both")
        update_real_torque = self._torque_mode in ("real", "both")
        if self._torque_mode != "off":
            for abbr, _usd_joint_name, _child_link_path in HAND_JOINT_TORQUE_RING_TUPLES:
                pair = self._torque_ring_prims.get(abbr)
                if not pair:
                    continue

                if update_sim_torque:
                    sim_val = 0.0
                    idx = self._abbr_to_dof_idx.get(abbr)
                    if sim_torques is not None and idx is not None:
                        try:
                            sim_val = float(sim_torques[idx])
                        except Exception:
                            sim_val = 0.0
                    self._apply_scale(pair.get("sim"), sim_val, TORQUE_GAIN,
                                      TORQUE_MIN_SCALE, TORQUE_MAX_SCALE)

                if update_real_torque:
                    real_val = float(real_snapshot.get(abbr, 0.0))
                    self._apply_scale(pair.get("real"), real_val, TORQUE_GAIN,
                                      TORQUE_MIN_SCALE, TORQUE_MAX_SCALE)

        # --- force arrows ---
        for entry in FORCE_VIZ_PARENT_LINKS:
            link_path = entry["link_path"]
            pair = self._force_viz_prims.get(link_path)
            if not pair:
                continue

            sim_mag = 0.0
            if sim_forces is not None:
                # sim_forces shape: (num_joint + 1, 6). row = joint_idx + 1.
                row = self._link_to_force_row.get(link_path)
                if row is not None and 0 <= row < len(sim_forces):
                    try:
                        f = sim_forces[row]
                        # 첫 3성분이 선형 힘
                        fx, fy, fz = float(f[0]), float(f[1]), float(f[2])
                        sim_mag = math.sqrt(fx * fx + fy * fy + fz * fz)
                    except Exception:
                        sim_mag = 0.0
            self._apply_force_scale(pair.get("sim"), sim_mag, FORCE_GAIN,
                                    FORCE_MIN_SCALE, FORCE_MAX_SCALE)

            # real force 는 현재 데이터 없음. 0 으로 표기.
            # TODO: real force/torque 토픽 수신 붙이면 abs(magnitude) 적용.
            self._apply_force_scale(pair.get("real"), 0.0, FORCE_GAIN,
                                    FORCE_MIN_SCALE, FORCE_MAX_SCALE)

    def set_mode(self, mode: str):
        """mode in {off, real, sim, both}."""
        if mode not in ("off", "real", "sim", "both"):
            logger.warning(f"[viz] unknown mode: {mode}")
            return
        self._mode = mode
        self._apply_visibility()

    def get_mode(self) -> str:
        return self._mode

    def set_torque_mode(self, mode: str):
        """mode in {off, real, sim, both}."""
        if mode not in ("off", "real", "sim", "both"):
            logger.warning(f"[viz] unknown torque mode: {mode}")
            return
        self._torque_mode = mode
        self._apply_visibility()

    def get_torque_mode(self) -> str:
        return self._torque_mode

    def set_torque_visibility(self, visible: bool):
        """하위호환 — True → both, False → off."""
        self.set_torque_mode("both" if visible else "off")

    def cleanup(self):
        """생성한 prim 들을 stage 에서 **제거**하고 캐시 해제 (B7).

        원본 `force_viz` (ALLEX.usd 에 원래 있던 real 화살표) 는 건드리지 않고,
        우리가 만든 `force_viz_sim`, `torque_ring_real`, `torque_ring_sim` 만 삭제한다.
        """
        try:
            stage = self._get_stage()
            if stage is not None:
                for path in list(self._created_prim_paths):
                    try:
                        prim = stage.GetPrimAtPath(path)
                        if prim and prim.IsValid():
                            stage.RemovePrim(path)
                    except Exception as e:
                        logger.debug(f"[viz] RemovePrim warn ({path}): {e}")
        except Exception as e:
            logger.debug(f"[viz] cleanup warn: {e}")

        self._torque_ring_prims.clear()
        self._force_viz_prims.clear()
        self._created_prim_paths.clear()
        self._abbr_to_dof_idx.clear()
        self._link_to_force_row.clear()
        self._scale_op_cache.clear()
        self._last_scale_cache.clear()
        self._force_length_cache.clear()
        self._prims_initialized = False

    # ========================================
    # Internal — prim 생성
    # ========================================
    def _get_stage(self):
        if self._stage is not None:
            return self._stage
        try:
            import omni.usd
            return omni.usd.get_context().get_stage()
        except Exception:
            return None

    def _ensure_prims(self):
        """토크 링 + sim force 화살표 prim 생성 (+ real force 캐시)."""
        from pxr import Gf
        stage = self._get_stage()
        if stage is None:
            logger.warning("[viz] stage not available, skip _ensure_prims")
            return

        torque_usd_exists = os.path.exists(TORQUE_VIZ_USD_PATH)
        force_usd_exists = os.path.exists(FORCE_VIZ_USD_PATH)
        if not torque_usd_exists:
            logger.warning(f"[viz] torque_viz.usd not found: {TORQUE_VIZ_USD_PATH}")
        if not force_usd_exists:
            logger.warning(f"[viz] force_viz.usd not found: {FORCE_VIZ_USD_PATH}")

        # ---- torque rings ----
        real_created = 0
        sim_created = 0
        for entry in HAND_JOINT_TORQUE_RING_MAP:
            parent_path = entry["child_link_path"]
            parent_prim = stage.GetPrimAtPath(parent_path)
            if not parent_prim or not parent_prim.IsValid():
                msg = f"[viz] parent link missing for torque ring: {parent_path}"
                logger.warning(msg)
                carb.log_warn(msg)
                continue

            real_path = f"{parent_path}/{TORQUE_RING_REAL_NAME}"
            sim_path = f"{parent_path}/{TORQUE_RING_SIM_NAME}"

            # Ring 의 local +Z 축을 joint rotation axis 에 정렬
            translate, orient_quat = self._compute_ring_transform(stage, entry["usd_joint_name"])
            base_scale = entry.get("base_scale", (1, 1, 1))

            # Real ring — REAL_COLOR
            real_prim = self._create_ref_xform(
                stage, real_path, TORQUE_VIZ_USD_PATH if torque_usd_exists else None,
                translate, orient_quat, base_scale, REAL_COLOR,
            )
            if real_prim is not None and real_prim.IsValid():
                real_created += 1
            else:
                msg = f"[viz] real torque ring NOT created: {real_path}"
                logger.warning(msg)
                carb.log_warn(msg)

            # Sim ring — SIM_COLOR. 경로가 real 과 다르므로 HasAuthoredReferences 가
            # 독립적으로 False → AddReference 수행됨.
            sim_prim = self._create_ref_xform(
                stage, sim_path, TORQUE_VIZ_USD_PATH if torque_usd_exists else None,
                translate, orient_quat, base_scale, SIM_COLOR,
            )
            if sim_prim is not None and sim_prim.IsValid():
                sim_created += 1
            else:
                msg = f"[viz] sim torque ring NOT created: {sim_path}"
                logger.warning(msg)
                carb.log_warn(msg)

            self._torque_ring_prims[entry["dof_abbr"]] = {
                "real": real_prim,
                "sim": sim_prim,
            }
        _torque_msg = (
            f"[viz] torque rings created — real={real_created}, sim={sim_created} "
            f"(expected {len(HAND_JOINT_TORQUE_RING_MAP)} each)"
        )
        print(_torque_msg)
        
        # ---- force arrows: real 은 기존, sim 은 신규 ----
        for entry in FORCE_VIZ_PARENT_LINKS:
            link_path = entry["link_path"]
            visuals_path = f"{link_path}/{FORCE_VIZ_VISUAL_SUBPATH}"
            real_path = f"{visuals_path}/{FORCE_VIZ_REAL_NAME}"
            sim_path = f"{visuals_path}/{FORCE_VIZ_SIM_NAME}"

            visuals_prim = stage.GetPrimAtPath(visuals_path)
            if not visuals_prim or not visuals_prim.IsValid():
                logger.warning(f"[viz] visuals scope missing: {visuals_path}")
                continue

            # Force 방향: entry 의 orient_euler_deg (link local frame 기준,
            # force_vec.usda 는 +Z 방향 화살표).
            orient_euler_deg = entry.get("orient_euler_deg", (0.0, 0.0, 0.0))
            fv_orient = self._euler_deg_to_quatd(orient_euler_deg)

            # Real prim: ALLEX.usd 가 sublayer 로 `force_viz` 를 old 구조 (Shaft/Head 없음)
            # 로 정의해 두었을 수 있다. 이 경로 prim 을 새로 fresh 하게 force_vec.usda 로
            # override 하기 위해 explicit Sdf.Reference 리스트를 사용한다.
            # translate 만 기존 prim 에서 읽어 유지 (link-local 원점 위치 보존).
            # Translate 기준: 기존 ALLEX.usd 의 force_viz prim (= sim_path) 에서 읽어옴
            existing_sim_prim = stage.GetPrimAtPath(sim_path)
            if existing_sim_prim and existing_sim_prim.IsValid():
                base_translate = self._read_translate(existing_sim_prim)
                if base_translate is None:
                    base_translate = Gf.Vec3d(0, 0, 0)
                logger.debug(
                    f"[viz] existing force_viz (sim) translate @ {sim_path}: "
                    f"{base_translate}"
                )
            else:
                base_translate = Gf.Vec3d(0, 0, 0)

            # Real prim: 신규 생성 (force_viz_real), REAL_COLOR
            real_prim = self._create_force_ref_xform(
                stage, real_path, FORCE_VIZ_USD_PATH if force_usd_exists else None,
                base_translate, fv_orient, (1, 1, 1), REAL_COLOR,
            )
            if real_prim is not None and force_usd_exists:
                self._cache_force_length_ops(real_prim, real_path)

            # Sim prim: 기존 force_viz prim override, SIM_COLOR
            sim_prim = self._create_force_ref_xform(
                stage, sim_path, FORCE_VIZ_USD_PATH if force_usd_exists else None,
                base_translate, fv_orient, (1, 1, 1), SIM_COLOR,
            )
            if sim_prim is not None and force_usd_exists:
                self._cache_force_length_ops(sim_prim, sim_path)

            self._force_viz_prims[link_path] = {
                "real": real_prim,
                "sim": sim_prim,
            }

        # contact sensor skeleton stub 호출 (TODO)
        for entry in FORCE_VIZ_PARENT_LINKS:
            self._create_contact_sensor_skeleton(entry["link_path"])

        # 초기 visibility 적용
        self._apply_visibility()

    def _create_ref_xform(self, stage, prim_path, usd_path,
                          translate, orient_quat, scale, color):
        """Xform prim 을 DefinePrim 으로 만들고 USD reference / xform op / display color 부여.

        - 참조 USD (torque_viz.usd, force_viz.usd) 가 double3/quatd precision 을 쓰므로
          모든 xformOp 을 PrecisionDouble 로 통일 (`PrecisionFloat` mismatch warning 회피).
        - orient_quat: Gf.Quatd 또는 (w, x, y, z) 튜플. None 이면 identity.

        B1 idempotency: 이미 authored reference 가 있으면 AddReference skip,
        이미 같은 displayColor 면 Set skip.
        """
        from pxr import UsdGeom, Gf
        try:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                prim = stage.DefinePrim(prim_path, "Xform")

            # 우리가 만든(또는 유지하는) prim path 로 등록 → cleanup 때 RemovePrim 대상
            self._created_prim_paths.add(prim_path)

            if usd_path is not None:
                try:
                    if not prim.HasAuthoredReferences():
                        refs = prim.GetReferences()
                        refs.AddReference(usd_path)
                    # 이미 reference 가 있으면 중복 추가하지 않음 (B1)
                except Exception as e:
                    logger.debug(f"[viz] AddReference warn ({prim_path}): {e}")

                # 참조 USD 가 갖고 있는 Physics 관련 prim/API 를 비활성화.
                # - torque_viz.usd 에는 `root_joint` (PhysicsFixedJoint + ArticulationRootAPI)
                #   가 있고, 이를 ALLEX link 하위에 ref 하면 PhysX 가 body0/body1 언-authored
                #   상태로 CreateJoint 에러 (40*2=80회) 를 낸다. visualization 전용이므로
                #   SetActive(False) 로 stage 에서 건너뛰게 만든다.
                # - 내부 RigidBody/ArticulationRoot 도 ALLEX articulation 하위에 들어가면
                #   혼란을 일으키므로 방어적으로 비활성화.
                self._disable_embedded_physics(stage, prim_path)

            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()

            # Translate (double)
            tv = Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2]))
            if any(abs(c) > 1e-12 for c in (tv[0], tv[1], tv[2])):
                xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(tv)

            # Orient (quatd)
            if orient_quat is not None:
                if isinstance(orient_quat, Gf.Quatd):
                    q = orient_quat
                elif isinstance(orient_quat, Gf.Quatf):
                    q = Gf.Quatd(float(orient_quat.GetReal()),
                                 Gf.Vec3d(*[float(x) for x in orient_quat.GetImaginary()]))
                else:
                    q = Gf.Quatd(float(orient_quat[0]),
                                 Gf.Vec3d(float(orient_quat[1]),
                                          float(orient_quat[2]),
                                          float(orient_quat[3])))
                # identity 면 skip
                im = q.GetImaginary()
                if not (abs(q.GetReal() - 1.0) < 1e-9
                        and abs(im[0]) < 1e-9 and abs(im[1]) < 1e-9 and abs(im[2]) < 1e-9):
                    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(q)

            # Scale (double — 참조 USD가 double3)
            scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
            s_init = Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2]))
            scale_op.Set(s_init)
            # scale op 캐시 (B5)
            self._scale_op_cache[prim_path] = scale_op
            self._last_scale_cache[prim_path] = float(scale[0])

            self._apply_display_color(prim, color)
            # torque_viz.usd 도 Looks/material_10000 OmniPBR 를 갖고 있으므로
            # displayColor 는 무시됨 → diffuse_color_constant 도 override
            if usd_path is not None:
                self._override_omni_pbr_color(stage, prim_path, color)
            return prim
        except Exception as e:
            logger.warning(f"[viz] create_ref_xform failed for {prim_path}: {e}")
            return None

    def _override_omni_pbr_color(self, stage, prim_path, color):
        """prim 하위 Looks/material_10000/Shader 의 diffuse_color_constant 를 override.

        torque_viz.usd 구조: <root>/Looks/material_10000/Shader (OmniPBR).
        이미 참조가 로드된 뒤에 호출해야 shader prim 이 존재한다.
        """
        from pxr import UsdShade, Sdf, Gf
        try:
            target_color = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
            # material_10000 고정 경로 먼저 시도
            shader_path = f"{prim_path}/Looks/material_10000/Shader"
            shader_prim = stage.GetPrimAtPath(shader_path)
            if not shader_prim or not shader_prim.IsValid():
                # fallback: Looks 하위 첫 번째 Shader 탐색
                looks_prim = stage.GetPrimAtPath(f"{prim_path}/Looks")
                if looks_prim and looks_prim.IsValid():
                    for mat_prim in looks_prim.GetChildren():
                        for child in mat_prim.GetChildren():
                            if child.GetTypeName() == "Shader":
                                shader_prim = child
                                break
                        if shader_prim and shader_prim.IsValid():
                            break
            if not shader_prim or not shader_prim.IsValid():
                return
            shader = UsdShade.Shader(shader_prim)
            inp = shader.GetInput("diffuse_color_constant")
            if not inp:
                inp = shader.CreateInput(
                    "diffuse_color_constant",
                    Sdf.ValueTypeNames.Color3f,
                )
            inp.Set(target_color)
        except Exception as e:
            logger.debug(f"[viz] override_omni_pbr_color warn ({prim_path}): {e}")

    def _create_force_ref_xform(self, stage, prim_path, usd_path,
                                translate, orient_quat, scale, color):
        """`_create_ref_xform` 과 거의 동일하지만, 모든 USD 조작을 **session layer**
        (최강 레이어)에 기록하여 ALLEX.usd sublayer 의 기존 reference / xformOp 을
        확실히 override 한다. `Usd.EditContext` 를 사용해 edit target 을 일시 전환.
        """
        from pxr import UsdGeom, Gf, Sdf, Usd
        try:
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                prim = stage.GetPrimAtPath(prim_path)
                if not prim or not prim.IsValid():
                    prim = stage.DefinePrim(prim_path, "Xform")

                self._created_prim_paths.add(prim_path)

                if usd_path is not None:
                    try:
                        refs = prim.GetReferences()
                        refs.SetReferences([Sdf.Reference(usd_path)])
                        # recomposition 후 prim 재조회
                        prim = stage.GetPrimAtPath(prim_path)
                        if not prim or not prim.IsValid():
                            msg = f"[viz] prim invalid after SetReferences: {prim_path}"
                            logger.warning(msg)
                            carb.log_warn(msg)
                            return None
                    except Exception as e:
                        msg = f"[viz] SetReferences warn ({prim_path}): {e}"
                        logger.debug(msg)
                        carb.log_warn(msg)

                    # 참조 USD 내부 Physics 비활성화
                    self._disable_embedded_physics(stage, prim_path)

                    # Mesh "visuals" (ALLEX.usd 인라인 geometry) 만 비활성화.
                    # Looks 스코프는 유지 — OmniPBR 재질을 Shaft/Head 에 재사용.
                    _legacy_mesh = stage.GetPrimAtPath(f"{prim_path}/visuals")
                    if _legacy_mesh and _legacy_mesh.IsValid():
                        try:
                            _legacy_mesh.SetActive(False)
                        except Exception:
                            pass

                xformable = UsdGeom.Xformable(prim)
                xformable.ClearXformOpOrder()

                # Translate (double)
                tv = Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2]))
                if any(abs(c) > 1e-12 for c in (tv[0], tv[1], tv[2])):
                    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(tv)

                # Orient (quatd)
                if orient_quat is not None:
                    if isinstance(orient_quat, Gf.Quatd):
                        q = orient_quat
                    elif isinstance(orient_quat, Gf.Quatf):
                        q = Gf.Quatd(float(orient_quat.GetReal()),
                                     Gf.Vec3d(*[float(x) for x in orient_quat.GetImaginary()]))
                    else:
                        q = Gf.Quatd(float(orient_quat[0]),
                                     Gf.Vec3d(float(orient_quat[1]),
                                              float(orient_quat[2]),
                                              float(orient_quat[3])))
                    im = q.GetImaginary()
                    if not (abs(q.GetReal() - 1.0) < 1e-9
                            and abs(im[0]) < 1e-9 and abs(im[1]) < 1e-9 and abs(im[2]) < 1e-9):
                        xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(q)

                # Scale (double)
                scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
                s_init = Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2]))
                scale_op.Set(s_init)
                self._scale_op_cache[prim_path] = scale_op
                self._last_scale_cache[prim_path] = float(scale[0])

                if usd_path is not None:
                    self._bind_omni_pbr_to_force_vec(stage, prim_path, color)
                else:
                    self._apply_display_color(prim, color)
                return prim
        except Exception as e:
            logger.warning(f"[viz] create_force_ref_xform failed for {prim_path}: {e}")
            return None

    def _bind_omni_pbr_to_force_vec(self, stage, prim_path, color):
        """Shaft/Head 메쉬에 OmniPBR 재질을 적용·바인딩한다.

        - 기존 Looks/material_10000 Shader 가 있으면 diffuse_color_constant 만 session
          layer 에서 override (빛 반사 등 기존 PBR 품질 유지).
        - 없으면 Looks/material_0 에 OmniPBR 재질을 신규 생성.
        - Shaft/Head 양쪽에 MaterialBindingAPI.Bind() 적용.

        색상 규칙: REAL_COLOR=(1,0,0) 빨강, SIM_COLOR=(0,0.8,1) 시안.
        """
        from pxr import UsdShade, Sdf, Gf
        try:
            target_color = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))

            # --- 재질 경로 결정: 기존 재사용 vs 신규 생성 ---
            existing_shader_prim = stage.GetPrimAtPath(
                f"{prim_path}/Looks/material_10000/Shader"
            )
            if existing_shader_prim and existing_shader_prim.IsValid():
                mat_path = f"{prim_path}/Looks/material_10000"
                shader_prim = existing_shader_prim
            else:
                mat_path = f"{prim_path}/Looks/material_0"
                looks_path = f"{prim_path}/Looks"
                if not stage.GetPrimAtPath(looks_path).IsValid():
                    stage.DefinePrim(looks_path, "Scope")
                mat = UsdShade.Material.Define(stage, mat_path)
                shader_p = stage.DefinePrim(f"{mat_path}/Shader", "Shader")
                shader_p.CreateAttribute(
                    "info:implementationSource", Sdf.ValueTypeNames.Token, True
                ).Set("sourceAsset")
                shader_p.CreateAttribute(
                    "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset, True
                ).Set(Sdf.AssetPath("OmniPBR.mdl"))
                shader_p.CreateAttribute(
                    "info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token, True
                ).Set("OmniPBR")
                shader_out = UsdShade.Shader(shader_p).CreateOutput(
                    "out", Sdf.ValueTypeNames.Token
                )
                mat.CreateOutput("mdl:surface", Sdf.ValueTypeNames.Token).ConnectToSource(
                    shader_out
                )
                mat.CreateOutput("mdl:displacement", Sdf.ValueTypeNames.Token).ConnectToSource(
                    shader_out
                )
                mat.CreateOutput("mdl:volume", Sdf.ValueTypeNames.Token).ConnectToSource(
                    shader_out
                )
                shader_prim = shader_p

            # --- 색상 override ---
            shader = UsdShade.Shader(shader_prim)
            inp = shader.GetInput("diffuse_color_constant")
            if not inp:
                inp = shader.CreateInput(
                    "diffuse_color_constant", Sdf.ValueTypeNames.Color3f
                )
            inp.Set(target_color)

            # --- Shaft / Head 에 재질 바인딩 ---
            mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))
            if not mat:
                return
            for child_name in (FORCE_VEC_SHAFT_NAME, FORCE_VEC_HEAD_NAME):
                child_prim = stage.GetPrimAtPath(f"{prim_path}/{child_name}")
                if child_prim and child_prim.IsValid():
                    try:
                        UsdShade.MaterialBindingAPI.Apply(child_prim).Bind(mat)
                    except Exception as bind_e:
                        logger.debug(
                            f"[viz] material bind warn ({child_name}@{prim_path}): {bind_e}"
                        )
        except Exception as e:
            logger.debug(f"[viz] bind_omni_pbr_to_force_vec warn ({prim_path}): {e}")

    def _compute_ring_transform(self, stage, joint_name):
        """Joint prim 을 읽어 torque ring 의 local translate + orient(quat) 계산.

        Ring USD (torque_viz.usd) 는 local +Z 축이 회전 대칭축.
        - translate: joint 의 localPos1 (body1=child_link 기준 joint 원점 위치)
        - orient:  +Z → axis_in_body1 으로 회전하는 quaternion.
                   axis_in_body1 = localRot1 · (axis_joint)
                   axis_joint = joint 의 axis attribute (X/Y/Z)
        """
        from pxr import UsdPhysics, Gf
        identity = (Gf.Vec3d(0, 0, 0), Gf.Quatd(1, 0, 0, 0))
        try:
            joint_prim = stage.GetPrimAtPath(f"/ALLEX/joints/{joint_name}")
            if not joint_prim or not joint_prim.IsValid():
                logger.debug(f"[viz] joint prim missing: /ALLEX/joints/{joint_name}")
                return identity
            rj = UsdPhysics.RevoluteJoint(joint_prim)
            if not rj:
                return identity

            axis_token = rj.GetAxisAttr().Get() or "X"
            axis_joint = {
                "X": Gf.Vec3d(1, 0, 0),
                "Y": Gf.Vec3d(0, 1, 0),
                "Z": Gf.Vec3d(0, 0, 1),
            }.get(axis_token, Gf.Vec3d(1, 0, 0))

            lp1 = rj.GetLocalPos1Attr().Get()
            lr1 = rj.GetLocalRot1Attr().Get()
            if lp1 is None:
                translate = Gf.Vec3d(0, 0, 0)
            else:
                translate = Gf.Vec3d(float(lp1[0]), float(lp1[1]), float(lp1[2]))

            if lr1 is None:
                axis_body1 = axis_joint
            else:
                lr1_d = Gf.Quatd(float(lr1.GetReal()),
                                 Gf.Vec3d(*[float(x) for x in lr1.GetImaginary()]))
                rot = Gf.Rotation(lr1_d)
                axis_body1 = Gf.Vec3d(rot.TransformDir(axis_joint))

            n = axis_body1.GetLength()
            if n < 1e-12:
                return (translate, Gf.Quatd(1, 0, 0, 0))
            axis_body1 = axis_body1 / n

            from_vec = Gf.Vec3d(0, 0, 1)
            d = Gf.Dot(from_vec, axis_body1)
            if d > 1.0 - 1e-9:
                orient = Gf.Quatd(1, 0, 0, 0)
            elif d < -1.0 + 1e-9:
                # 180° about X (arbitrary orthogonal axis)
                orient = Gf.Quatd(0.0, Gf.Vec3d(1, 0, 0))
            else:
                orient = Gf.Rotation(from_vec, axis_body1).GetQuat()

            return (translate, orient)
        except Exception as e:
            logger.debug(f"[viz] compute_ring_transform warn ({joint_name}): {e}")
            return identity

    def _cache_scale_op(self, prim, prim_path):
        """기존 prim(예: 원본 real force_viz) 의 scale op 를 찾아 캐시."""
        try:
            from pxr import UsdGeom
            xformable = UsdGeom.Xformable(prim)
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    self._scale_op_cache[prim_path] = op
                    return
            # scale op 이 아예 없으면 하나 추가
            scale_op = xformable.AddScaleOp()
            self._scale_op_cache[prim_path] = scale_op
        except Exception as e:
            logger.debug(f"[viz] cache_scale_op warn ({prim_path}): {e}")

    def _apply_display_color(self, prim, color):
        """Gprim 자식에 displayColor 적용. B1: 이미 같은 색이면 skip."""
        try:
            from pxr import UsdGeom, Gf, Vt
            target_color = Gf.Vec3f(*color)

            def _set_if_changed(attr):
                try:
                    existing = attr.Get()
                    if existing and len(existing) == 1:
                        c = existing[0]
                        if (abs(c[0] - target_color[0]) < 1e-4 and
                                abs(c[1] - target_color[1]) < 1e-4 and
                                abs(c[2] - target_color[2]) < 1e-4):
                            return
                except Exception:
                    pass
                attr.Set(Vt.Vec3fArray([target_color]))

            gprim = UsdGeom.Gprim(prim)
            if gprim:
                attr = gprim.GetDisplayColorAttr()
                if attr:
                    _set_if_changed(attr)
                    return
            # 자식 순회
            for child in prim.GetChildren():
                child_g = UsdGeom.Gprim(child)
                if child_g:
                    attr = child_g.GetDisplayColorAttr()
                    if attr:
                        _set_if_changed(attr)
        except Exception as e:
            logger.debug(f"[viz] apply_display_color warn: {e}")

    def _disable_embedded_physics(self, stage, prim_path):
        """참조된 visualization USD 안의 Physics 관련 요소 비활성화.

        - `PhysicsJoint` 타입 prim (예: `{prim_path}/root_joint`) 은 SetActive(False).
          -> PhysX 가 body0/body1 언-authored CreateJoint 에러를 내는 원인 제거.
        - PhysicsRigidBodyAPI / PhysicsArticulationRootAPI / PhysicsCollisionAPI /
          PhysxArticulationAPI 가 붙은 prim 은 RemoveAppliedSchema 로 스키마만 제거.
          (visual mesh 자식을 숨기지 않기 위해 SetActive 는 사용하지 않는다.)
        """
        try:
            root = stage.GetPrimAtPath(prim_path)
            if not root or not root.IsValid():
                return
            physics_joint_types = (
                "PhysicsJoint",
                "PhysicsFixedJoint",
                "PhysicsRevoluteJoint",
                "PhysicsPrismaticJoint",
                "PhysicsSphericalJoint",
                "PhysicsDistanceJoint",
            )
            physics_api_names = (
                "PhysicsRigidBodyAPI",
                "PhysicsArticulationRootAPI",
                "PhysxArticulationAPI",
                "PhysicsCollisionAPI",
                "PhysicsMassAPI",
            )
            for p in Usd_iter_descendants(root):
                try:
                    ptype = p.GetTypeName()
                    if ptype in physics_joint_types:
                        p.SetActive(False)
                        continue
                    apis = p.GetAppliedSchemas() or []
                    for api_name in physics_api_names:
                        if api_name in apis:
                            try:
                                p.RemoveAppliedSchema(api_name)
                            except Exception as e2:
                                logger.debug(
                                    f"[viz] RemoveAppliedSchema({api_name}) warn "
                                    f"on {p.GetPath().pathString}: {e2}"
                                )
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"[viz] disable_embedded_physics warn ({prim_path}): {e}")

    def _apply_orient_to_existing(self, prim, orient_quat):
        """기존 prim (ref 로 만들지 않은 원본 prim 예: real force_viz) 에
        orient quaternion 을 반영. 기존 xform op 순서를 최대한 보존하고,
        orient op 가 이미 있으면 Set, 없으면 AddOrientOp 를 추가한다.
        """
        if prim is None or orient_quat is None:
            return
        try:
            from pxr import UsdGeom, Gf
            if not prim.IsValid():
                return

            # identity 면 skip
            if isinstance(orient_quat, Gf.Quatd):
                q = orient_quat
            elif isinstance(orient_quat, Gf.Quatf):
                q = Gf.Quatd(float(orient_quat.GetReal()),
                             Gf.Vec3d(*[float(x) for x in orient_quat.GetImaginary()]))
            else:
                q = Gf.Quatd(float(orient_quat[0]),
                             Gf.Vec3d(float(orient_quat[1]),
                                      float(orient_quat[2]),
                                      float(orient_quat[3])))
            im = q.GetImaginary()
            if (abs(q.GetReal() - 1.0) < 1e-9
                    and abs(im[0]) < 1e-9 and abs(im[1]) < 1e-9 and abs(im[2]) < 1e-9):
                return

            xformable = UsdGeom.Xformable(prim)
            existing_orient_op = None
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    existing_orient_op = op
                    break

            if existing_orient_op is not None:
                try:
                    if existing_orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
                        existing_orient_op.Set(q)
                    else:
                        existing_orient_op.Set(Gf.Quatf(
                            float(q.GetReal()),
                            Gf.Vec3f(float(im[0]), float(im[1]), float(im[2])),
                        ))
                except Exception as e:
                    logger.debug(f"[viz] set existing orient warn: {e}")
            else:
                try:
                    # 기존 op 순서 맨 뒤에 추가 (translate 는 유지, scale 앞)
                    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(q)
                except Exception as e:
                    logger.debug(f"[viz] add orient op warn: {e}")
        except Exception as e:
            logger.debug(f"[viz] apply_orient_to_existing warn: {e}")

    def _euler_deg_to_quatd(self, euler_xyz_deg):
        """(rx, ry, rz) [deg] → Gf.Quatd (XYZ intrinsic 결합).

        결합 순서: R = Rx * Ry * Rz (vector 에 적용 시 Rz 먼저, 다음 Ry, 다음 Rx —
        intrinsic XYZ 와 동치).
        """
        try:
            from pxr import Gf
            rx_deg = float(euler_xyz_deg[0])
            ry_deg = float(euler_xyz_deg[1])
            rz_deg = float(euler_xyz_deg[2])
            rx = Gf.Rotation(Gf.Vec3d(1, 0, 0), rx_deg)
            ry = Gf.Rotation(Gf.Vec3d(0, 1, 0), ry_deg)
            rz = Gf.Rotation(Gf.Vec3d(0, 0, 1), rz_deg)
            total = rx * ry * rz
            q = total.GetQuat()
            # Gf.Rotation.GetQuat() → Gf.Quaternion. Gf.Quatd 로 명시 변환.
            try:
                if isinstance(q, Gf.Quatd):
                    qd = q
                else:
                    qd = Gf.Quatd(float(q.GetReal()),
                                  Gf.Vec3d(*[float(x) for x in q.GetImaginary()]))
            except Exception:
                qd = Gf.Quatd(1, 0, 0, 0)
            logger.debug(
                f"[viz] euler{tuple(euler_xyz_deg)}deg → quat(w,x,y,z)="
                f"({qd.GetReal():.4f}, {qd.GetImaginary()[0]:.4f}, "
                f"{qd.GetImaginary()[1]:.4f}, {qd.GetImaginary()[2]:.4f})"
            )
            return qd
        except Exception as e:
            logger.debug(f"[viz] euler_deg_to_quatd warn: {e}")
            from pxr import Gf
            return Gf.Quatd(1, 0, 0, 0)

    def _read_translate(self, prim):
        """prim 의 xformOp:translate 값을 Gf.Vec3d 로 읽어 반환. 없으면 None."""
        if prim is None:
            return None
        try:
            from pxr import UsdGeom, Gf
            if not prim.IsValid():
                return None
            xformable = UsdGeom.Xformable(prim)
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    v = op.Get()
                    if v is None:
                        return None
                    return Gf.Vec3d(float(v[0]), float(v[1]), float(v[2]))
            return None
        except Exception as e:
            logger.debug(f"[viz] read_translate warn: {e}")
            return None

    def _cache_force_length_ops(self, prim, prim_path):
        """force 화살표 prim 내부 Shaft/Head 의 scale/translate op 를 캐시.

        force_vec.usda 구조 (xformOps 원본에 없음):
          <prim_path>/Shaft  : Z=0..FORCE_VEC_SHAFT_BASE_LENGTH 원통 (radius 0.003)
          <prim_path>/Head   : Z=FORCE_VEC_SHAFT_BASE_LENGTH..0.009 원뿔

        대칭 grow 방식:
          Shaft  : scaleZ = s, translateZ = -L_base * s / 2
                   (shaft 는 -L*s/2 .. +L*s/2 에 걸쳐 원점 기준 대칭 확장)
          Head   : translateZ = L_base * (s - 2) / 2
                   (head 바닥이 shaft 상단 +L*s/2 에 정확히 얹힌다)

        s=1 초기값:
          Shaft  scale=(1,1,1), translate=(0,0,-L_base/2)
          Head   translate=(0,0,-L_base/2)
        """
        if prim is None:
            return
        try:
            from pxr import UsdGeom, Gf
            stage = self._get_stage()
            if stage is None:
                return

            shaft_path = f"{prim_path}/{FORCE_VEC_SHAFT_NAME}"
            head_path = f"{prim_path}/{FORCE_VEC_HEAD_NAME}"
            shaft_prim = stage.GetPrimAtPath(shaft_path)
            head_prim = stage.GetPrimAtPath(head_path)
            if (not shaft_prim or not shaft_prim.IsValid()
                    or not head_prim or not head_prim.IsValid()):
                logger.debug(f"[viz] force_vec shaft/head not found under {prim_path}")
                return

            L_base = float(FORCE_VEC_SHAFT_BASE_LENGTH)
            axis = FORCE_VEC_LENGTH_AXIS

            # --- Shaft: translate + scale (이 순서 — scale 먼저 적용, 그 다음 translate) ---
            shaft_xformable = UsdGeom.Xformable(shaft_prim)
            # 기존 op 제거 후 재구성 (순서 강제를 위해).
            shaft_xformable.ClearXformOpOrder()

            shaft_translate_op = shaft_xformable.AddTranslateOp(
                UsdGeom.XformOp.PrecisionDouble,
            )
            shaft_t_init = [0.0, 0.0, 0.0]
            shaft_t_init[axis] = -L_base / 2.0
            shaft_translate_op.Set(Gf.Vec3d(*shaft_t_init))

            shaft_scale_op = shaft_xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
            shaft_scale_op.Set(Gf.Vec3d(1.0, 1.0, 1.0))

            # --- Head: translate only ---
            head_xformable = UsdGeom.Xformable(head_prim)
            head_translate_op = None
            for op in head_xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    head_translate_op = op
                    break
            if head_translate_op is None:
                head_translate_op = head_xformable.AddTranslateOp(
                    UsdGeom.XformOp.PrecisionDouble,
                )
            head_t_init = [0.0, 0.0, 0.0]
            head_t_init[axis] = -L_base / 2.0  # s=1 → L_base*(1-2)/2 = -L_base/2
            head_translate_op.Set(Gf.Vec3d(*head_t_init))

            self._force_length_cache[prim_path] = {
                "shaft_scale_op": shaft_scale_op,
                "shaft_translate_op": shaft_translate_op,
                "head_translate_op": head_translate_op,
                "last_scale": None,
            }
        except Exception as e:
            logger.debug(f"[viz] cache_force_length_ops warn ({prim_path}): {e}")

    # ========================================
    # Internal — update helpers
    # ========================================
    def _build_abbr_to_dof_idx(self):
        """articulation.dof_names 에서 약어→idx 매핑 구축.

        USD joint prim name 과 dof name 은 동일하다고 가정 (ALLEX 기준 확인).
        entry 의 usd_joint_name 으로 strict equality 매칭만 허용 (S1).
        """
        self._abbr_to_dof_idx.clear()
        if self._articulation is None:
            return
        try:
            names = list(self._articulation.dof_names or [])
        except Exception as e:
            logger.warning(f"[viz] dof_names unavailable: {e}")
            return

        name_to_idx = {n: i for i, n in enumerate(names)}
        missing = []
        for entry in HAND_JOINT_TORQUE_RING_MAP:
            joint_name = entry["usd_joint_name"]
            idx = name_to_idx.get(joint_name)
            if idx is None:
                missing.append(joint_name)
            else:
                self._abbr_to_dof_idx[entry["dof_abbr"]] = idx
        if missing:
            logger.warning(
                f"[viz] {len(missing)} hand joint(s) not found in dof_names: {missing[:5]}..."
            )

    def _build_link_to_force_row(self):
        """link_path → get_measured_joint_forces row index LUT.

        row = joint_idx + 1  (row 0 은 base link incoming joint).
        joint_idx 는 articulation metadata 의 joint_indices 에서 조회.

        공식 Isaac Sim 예제에도 `prim._articulation_view._metadata.joint_indices`
        접근법이 소개되어 있으므로 사실상 semi-public. 실패 시 fallback 으로
        `articulation.body_names` 순서를 활용 시도.
        """
        self._link_to_force_row.clear()
        if self._articulation is None:
            return

        joint_name_to_idx = {}
        try:
            view = getattr(self._articulation, "_articulation_view", None)
            meta = getattr(view, "_metadata", None) if view is not None else None
            if meta is not None:
                ji = getattr(meta, "joint_indices", None)
                if ji:
                    joint_name_to_idx = dict(ji)
        except Exception as e:
            logger.debug(f"[viz] _metadata.joint_indices access failed: {e}")

        # Fallback: joint name 이 articulation.dof_names 순서와 일치하는 경우에만 유용.
        # (fixed joint 가 섞여 있으면 부정확하므로 실패 시 None 처리.)
        if not joint_name_to_idx:
            try:
                dof_names = list(self._articulation.dof_names or [])
                if dof_names:
                    joint_name_to_idx = {n: i for i, n in enumerate(dof_names)}
                    print(
                        "[viz] fallback: using dof_names as joint_indices for force row LUT "
                        "(may be inaccurate if fixed joints are interleaved)."
                    )
            except Exception:
                pass

        if not joint_name_to_idx:
            logger.warning("[viz] could not build link_to_force_row LUT — sim force will be 0")
            return

        # HAND entry 기반: child_link_path → usd_joint_name → joint_idx → row
        link_to_row = {}
        for entry in HAND_JOINT_TORQUE_RING_MAP:
            jn = entry["usd_joint_name"]
            idx = joint_name_to_idx.get(jn)
            if idx is not None:
                link_to_row[entry["child_link_path"]] = idx + 1

        # FORCE_VIZ_PARENT_LINKS 중 Palm 은 HAND entry 에 없음 → 정책: 0 으로 둠.
        # TODO: Palm 에 매핑할 wrist joint 결정되면 여기에 추가.
        #       예: "/ALLEX/L_Palm_Link" → "L_Wrist_Pitch_Joint" 등.
        for entry in FORCE_VIZ_PARENT_LINKS:
            lp = entry["link_path"]
            if lp not in link_to_row:
                # Palm 류: row 없음 → 조회 시 None → sim_mag=0
                continue

        self._link_to_force_row = link_to_row
        missing_force_links = [
            e["link_path"] for e in FORCE_VIZ_PARENT_LINKS
            if e["link_path"] not in link_to_row
        ]
        if missing_force_links:
            print(
                f"[viz] {len(missing_force_links)} force-viz link(s) without joint row "
                f"(Palm etc.): {missing_force_links}"
            )

    def _apply_scale(self, prim, raw_value, gain, lo, hi):
        """캐시된 scale op 에 직접 Set. dead-band 통과 시에만 write (B5)."""
        if prim is None:
            return
        try:
            prim_path = prim.GetPath().pathString
        except Exception:
            prim_path = None

        try:
            v = float(raw_value)
            mag = abs(v) * gain              # 크기만 gain 적용 (방향은 copysign 으로)
            s = math.copysign(_clip(mag, lo, hi), v) if v != 0.0 else lo

            # dead-band check
            if prim_path is not None:
                prev = self._last_scale_cache.get(prim_path)
                if prev is not None and abs(s - prev) < _SCALE_DEADBAND:
                    return

            scale_op = self._scale_op_cache.get(prim_path) if prim_path else None
            if scale_op is None:
                # cache miss — 복구 시도 (GetOrderedXformOps linear scan, 이후 캐시)
                from pxr import UsdGeom
                xformable = UsdGeom.Xformable(prim)
                for op in xformable.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        scale_op = op
                        break
                if scale_op is None:
                    scale_op = xformable.AddScaleOp()
                if prim_path is not None:
                    self._scale_op_cache[prim_path] = scale_op

            from pxr import Gf, UsdGeom
            # precision 에 맞춰 Vec3 타입 결정 (double3 vs float3)
            try:
                is_double = scale_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble
            except Exception:
                is_double = True
            if is_double:
                scale_op.Set(Gf.Vec3d(s, s, s))
            else:
                scale_op.Set(Gf.Vec3f(s, s, s))
            if prim_path is not None:
                self._last_scale_cache[prim_path] = s
        except Exception as e:
            logger.debug(f"[viz] apply_scale warn: {e}")

    def _apply_force_scale(self, prim, raw_value, gain, lo, hi):
        """force 화살표 전용: Shaft 길이(Z scale) + Head translate Z 를 함께 갱신.

        shaft/head op 캐시가 없으면 legacy `_apply_scale` 로 fallback (전체 scale).
        """
        if prim is None:
            return
        try:
            prim_path = prim.GetPath().pathString
        except Exception:
            prim_path = None

        cache = self._force_length_cache.get(prim_path) if prim_path else None
        if cache is None:
            # fallback — 전체 scale
            self._apply_scale(prim, raw_value, gain, lo, hi)
            return

        try:
            s = _clip(abs(float(raw_value)) * gain, lo, hi)
            prev = cache.get("last_scale")
            if prev is not None and abs(s - prev) < _SCALE_DEADBAND:
                return

            from pxr import Gf, UsdGeom
            shaft_scale_op = cache["shaft_scale_op"]
            shaft_translate_op = cache.get("shaft_translate_op")
            head_op = cache["head_translate_op"]

            axis = FORCE_VEC_LENGTH_AXIS
            L_base = float(FORCE_VEC_SHAFT_BASE_LENGTH)

            # --- Shaft scale (길이 축만 s, 나머지 1.0) ---
            scale_vec = [1.0, 1.0, 1.0]
            scale_vec[axis] = s
            try:
                is_double = shaft_scale_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble
            except Exception:
                is_double = True
            if is_double:
                shaft_scale_op.Set(Gf.Vec3d(*scale_vec))
            else:
                shaft_scale_op.Set(Gf.Vec3f(*scale_vec))

            # --- Shaft translate (대칭 중심화: -L_base*s/2) ---
            if shaft_translate_op is not None:
                shaft_t = [0.0, 0.0, 0.0]
                shaft_t[axis] = -L_base * s / 2.0
                try:
                    is_double_st = (shaft_translate_op.GetPrecision()
                                    == UsdGeom.XformOp.PrecisionDouble)
                except Exception:
                    is_double_st = True
                if is_double_st:
                    shaft_translate_op.Set(Gf.Vec3d(*shaft_t))
                else:
                    shaft_translate_op.Set(Gf.Vec3f(*shaft_t))

            # --- Head translate (shaft 상단에 바닥이 얹히도록: L_base*(s-2)/2) ---
            head_t = [0.0, 0.0, 0.0]
            head_t[axis] = L_base * (s - 2.0) / 2.0
            try:
                is_double_h = head_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble
            except Exception:
                is_double_h = True
            if is_double_h:
                head_op.Set(Gf.Vec3d(*head_t))
            else:
                head_op.Set(Gf.Vec3f(*head_t))

            cache["last_scale"] = s
        except Exception as e:
            logger.debug(f"[viz] apply_force_scale warn: {e}")

    def _apply_visibility(self):
        """force / torque 가시성 반영. 두 경로 독립.

        - torque rings: `_torque_mode` (off/real/sim/both)
        - force arrows: `_mode` (off/real/sim/both)
        """
        # --- torque rings ---
        show_real_torque = self._torque_mode in ("real", "both")
        show_sim_torque = self._torque_mode in ("sim", "both")
        for pair in self._torque_ring_prims.values():
            self._set_visible(pair.get("real"), show_real_torque)
            self._set_visible(pair.get("sim"), show_sim_torque)

        # --- force arrows ---
        show_real_force = self._mode in ("real", "both")
        show_sim_force = self._mode in ("sim", "both")
        for pair in self._force_viz_prims.values():
            self._set_visible(pair.get("real"), show_real_force)
            self._set_visible(pair.get("sim"), show_sim_force)

    def _set_visible(self, prim, visible):
        """prim 가시성 토글.

        UsdGeom.Imageable.MakeVisible/MakeInvisible 은 상위 Xform 의 invisible
        토큰까지 inherited 로 되돌리므로, 이전 세션에서 남은 visibility 상태를
        확실히 재설정할 수 있다 (문제 3의 원인 후보 중 하나).
        """
        if prim is None:
            return
        try:
            from pxr import UsdGeom
            if not prim.IsValid():
                return
            imageable = UsdGeom.Imageable(prim)
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        except Exception as e:
            logger.debug(f"[viz] set_visible warn: {e}")

    # ========================================
    # Contact sensor skeleton
    # ========================================
    def _create_contact_sensor_skeleton(self, link_path):
        """Force 화살표 부모 link 에 contact sensor 를 붙이기 위한 스텁.

        TODO: isaacsim.sensors.contact / PhysxContactReportAPI 를 사용해
              실제 contact force 를 읽어오고 real 화살표 크기에 반영.
              현재는 no-op.
        """
        return None
