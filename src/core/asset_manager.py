"""
ALLEX Digital Twin 에셋 관리
"""

from pathlib import Path
import numpy as np
import omni.usd
from pxr import UsdGeom, Gf
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage


# MJCF-intended drive gains (SI units: N·m/rad stiffness, N·m·s/rad damping).
# USD authored values are ~57× these (PhysX deg-convention), causing extreme
# actuator stiffness under Newton/MuJoCo. We override at runtime to restore
# physically-correct behavior.
_MJCF_DRIVE_GAINS: dict[str, tuple[float, float]] = {
    "Waist_Yaw_Joint":        (2000.0, 100.0),
    "Waist_Lower_Pitch_Joint": (600.0, 120.0),
    "Neck_Pitch_Joint":        (100.0,   5.0),
    "Neck_Yaw_Joint":          (100.0,   2.5),
    "L_Shoulder_Pitch_Joint": (5000.0, 200.0),
    "L_Shoulder_Roll_Joint":  (5000.0, 200.0),
    "L_Shoulder_Yaw_Joint":   (5000.0, 100.0),
    "L_Elbow_Joint":          (1000.0,  50.0),
    "L_Wrist_Yaw_Joint":       (400.0,  10.0),
    "L_Wrist_Roll_Joint":      (100.0,   5.0),
    "L_Wrist_Pitch_Joint":     (100.0,   5.0),
    "L_Thumb_Yaw_Joint":        (40.0,   1.0),
    "L_Thumb_CMC_Joint":        (40.0,   1.0),
    "L_Thumb_MCP_Joint":        (20.0,   0.5),
    "L_Index_ABAD_Joint":       (20.0,   0.5),
    "L_Index_MCP_Joint":        (40.0,   1.0),
    "L_Index_PIP_Joint":        (20.0,   0.5),
    "L_Little_ABAD_Joint":      (20.0,   0.5),
    "L_Little_MCP_Joint":       (40.0,   1.0),
    "L_Little_PIP_Joint":       (20.0,   0.5),
    "L_Middle_ABAD_Joint":      (20.0,   0.5),
    "L_Middle_MCP_Joint":       (40.0,   1.0),
    "L_Middle_PIP_Joint":       (20.0,   0.5),
    "L_Ring_ABAD_Joint":        (20.0,   0.5),
    "L_Ring_MCP_Joint":         (40.0,   1.0),
    "L_Ring_PIP_Joint":         (20.0,   0.5),
    "R_Shoulder_Pitch_Joint": (5000.0, 200.0),
    "R_Shoulder_Roll_Joint":  (5000.0, 200.0),
    "R_Shoulder_Yaw_Joint":   (5000.0, 100.0),
    "R_Elbow_Joint":          (1000.0,  50.0),
    "R_Wrist_Yaw_Joint":       (400.0,  10.0),
    "R_Wrist_Roll_Joint":      (100.0,   5.0),
    "R_Wrist_Pitch_Joint":     (100.0,   5.0),
    "R_Thumb_Yaw_Joint":        (40.0,   1.0),
    "R_Thumb_CMC_Joint":        (40.0,   1.0),
    "R_Thumb_MCP_Joint":        (20.0,   0.5),
    "R_Index_ABAD_Joint":       (20.0,   0.5),
    "R_Index_MCP_Joint":        (40.0,   1.0),
    "R_Index_PIP_Joint":        (20.0,   0.5),
    "R_Little_ABAD_Joint":      (20.0,   0.5),
    "R_Little_MCP_Joint":       (40.0,   1.0),
    "R_Little_PIP_Joint":       (20.0,   0.5),
    "R_Middle_ABAD_Joint":      (20.0,   0.5),
    "R_Middle_MCP_Joint":       (40.0,   1.0),
    "R_Middle_PIP_Joint":       (20.0,   0.5),
    "R_Ring_ABAD_Joint":        (20.0,   0.5),
    "R_Ring_MCP_Joint":         (40.0,   1.0),
    "R_Ring_PIP_Joint":         (20.0,   0.5),
}


def get_default_gains_vec(articulation) -> "tuple[np.ndarray, np.ndarray] | None":
    """Return current per-DOF (kp, kd) vectors from the articulation view.

    Shape is (num_dof,) numpy float32. Returns None on failure.
    """
    try:
        view = articulation._articulation_view
        kps, kds = view.get_gains()
    except Exception as exc:
        print(f"[ALLEX][Gains] get_gains failed: {exc}")
        return None

    def _to_np(t):
        if t is None:
            return None
        if hasattr(t, "detach"):
            t = t.detach().cpu().numpy()
        arr = np.asarray(t, dtype=np.float32).reshape(-1)
        return arr

    kp_vec = _to_np(kps)
    kd_vec = _to_np(kds)
    if kp_vec is None or kd_vec is None:
        return None
    return kp_vec, kd_vec


def _override_drive_gains(articulation) -> int:
    """Replace per-DOF drive stiffness/damping with MJCF-intended SI values."""
    try:
        view = articulation._articulation_view
    except Exception:
        return 0
    try:
        names = list(articulation.dof_names or [])
    except Exception:
        return 0
    num_dof = len(names)
    if num_dof == 0:
        return 0
    try:
        cur_kps, cur_kds = view.get_gains()
    except Exception as exc:
        print(f"[ALLEX][Gains] get_gains failed: {exc}")
        return 0
    try:
        import torch as _torch
        kps = cur_kps.clone() if hasattr(cur_kps, "clone") else _torch.tensor(cur_kps)
        kds = cur_kds.clone() if hasattr(cur_kds, "clone") else _torch.tensor(cur_kds)
    except Exception:
        import numpy as _np
        kps = _np.array(cur_kps).copy()
        kds = _np.array(cur_kds).copy()

    changed = 0
    for i, n in enumerate(names):
        gains = _MJCF_DRIVE_GAINS.get(n)
        if gains is None:
            continue
        kp, kd = gains
        # tensor or ndarray
        try:
            kps[0, i] = kp
            kds[0, i] = kd
        except Exception:
            try:
                kps[i] = kp
                kds[i] = kd
            except Exception:
                continue
        changed += 1

    try:
        view.set_gains(kps=kps, kds=kds)
        print(f"[ALLEX][Gains] overrode drive gains on {changed}/{num_dof} dof(s)")
    except Exception as exc:
        print(f"[ALLEX][Gains] set_gains failed: {exc}")
        return 0
    return changed


class ALLEXAssetManager:
    """ALLEX 디지털 트윈 에셋 로딩 및 관리를 담당하는 클래스"""
    
    ROBOT_SPAWN_POSITION = np.array([0.0, 0.0, 0.685])

    def __init__(self):
        """에셋 관리자 초기화"""
        self._articulation = None
        self._robot_prim_path = "/ALLEX"

    def load_robot_asset(self):
        """ALLEX 로봇 에셋 로딩
        
        Returns:
            SingleArticulation: 로딩된 로봇 Articulation 객체
        """
        try:
            current_file = Path(__file__)
            # core 폴더에서 상위로 이동하여 asset 폴더 찾기
            asset_path = current_file.parent.parent.parent / "asset" / "ALLEX" / "ALLEX.usd"
            path_to_robot_usd = str(asset_path.resolve())

            # 에셋이 존재하는지 확인
            if not asset_path.exists():
                raise FileNotFoundError(f"로봇 에셋 파일을 찾을 수 없습니다: {asset_path}")

            # Re-apply Newton cfg overrides from TOML just before the robot
            # is referenced into the stage — this ensures pd_scale and solver
            # settings are in place when Newton's add_usd() runs on first RUN.
            try:
                from . import newton_bridge as _nb
                _nb.configure_newton_from_toml()
            except Exception as exc:
                print(f"[ALLEX][Cfg] newton cfg apply failed: {exc}")

            # 스테이지에 로봇 에셋 추가
            add_reference_to_stage(path_to_robot_usd, self._robot_prim_path)

            # Apply USD /physicsScene overrides from config/physics.toml.
            try:
                stage_ctx = omni.usd.get_context().get_stage()
                from ..config import physics_settings as _ps
                _ps.apply_usd_physics_scene(stage_ctx)
            except Exception as exc:
                print(f"[ALLEX][Cfg] physics_scene apply failed: {exc}")

            # prim translate로 z 위치 직접 설정
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self._robot_prim_path)
            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.685))

            # Articulation 객체 생성 (초기화는 나중에)
            self._articulation = SingleArticulation(self._robot_prim_path)

            return self._articulation
            
        except Exception as e:
            print(f"❌ 로봇 에셋 로딩 실패: {e}")
            return None

    def initialize_articulation(self):
        """Articulation 초기화 (시뮬레이션 시작 후 호출)
        
        Returns:
            bool: 초기화 성공 여부
        """
        if self._articulation is None:
            print("⚠️ Articulation 객체가 없습니다")
            return False
            
        # Re-apply USD physicsScene config now that physics scene exists.
        try:
            stage_ctx = omni.usd.get_context().get_stage()
            from ..config import physics_settings as _ps
            _ps.apply_usd_physics_scene(stage_ctx)
        except Exception as exc:
            print(f"[ALLEX][Cfg] post-init physics_scene apply failed: {exc}")

        try:
            self._articulation.initialize()
            try:
                names = list(self._articulation.dof_names or [])
                print(f"[ALLEX][Art] initialize ok, num_dof={len(names)}")
                for i, n in enumerate(names):
                    print(f"   dof[{i:3d}] {n}")
            except Exception as exc:
                print(f"[ALLEX][Art] dof_names dump failed: {exc}")
            try:
                view = self._articulation._articulation_view  # underlying view has get_gains
                kps, kds = view.get_gains()
                def _to_list(t):
                    if t is None:
                        return []
                    if hasattr(t, "detach"):
                        t = t.detach().cpu().numpy()
                    import numpy as _np
                    return _np.asarray(t).reshape(-1).tolist()
                kp_list = _to_list(kps)
                kd_list = _to_list(kds)
                print(f"[ALLEX][Art] drive gains dump (stiffness, damping):")
                for i, n in enumerate(names):
                    kp = kp_list[i] if i < len(kp_list) else "?"
                    kd = kd_list[i] if i < len(kd_list) else "?"
                    print(f"   gain[{i:3d}] kp={kp} kd={kd}  ({n})")
            except Exception as exc:
                print(f"[ALLEX][Art] gains dump failed: {exc}")
            return True
        except Exception:
            return False

    def get_articulation(self):
        """현재 로딩된 Articulation 객체 반환
        
        Returns:
            SingleArticulation: 로봇 Articulation 객체
        """
        return self._articulation

    def get_robot_prim_path(self):
        """로봇 프림 경로 반환
        
        Returns:
            str: 로봇 프림 경로
        """
        return self._robot_prim_path

    def is_robot_loaded(self):
        """로봇이 로딩되었는지 확인
        
        Returns:
            bool: 로봇 로딩 상태
        """
        return self._articulation is not None

    def get_joint_info(self):
        """관절 정보 반환
        
        Returns:
            dict: 관절 관련 정보 (이름, 개수 등)
        """
        if self._articulation is None:
            return None
            
        try:
            joint_names = self._articulation.dof_names
            joint_count = len(joint_names) if joint_names else 0
            
            return {
                'joint_names': joint_names,
                'joint_count': joint_count,
                'articulation_path': self._robot_prim_path
            }
        except Exception as e:
            print(f"⚠️ 관절 정보 조회 실패: {e}")
            return None