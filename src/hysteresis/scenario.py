"""Hysteresis Scenario — ALLEX_Hand.usd 전용 핸드 단독 시나리오.

`ALLEXDigitalTwin` 을 축소 복제해 손만 로드한다. 공유 인프라
(`core.ALLEXSimulationLoop`, `core.ALLEXJointController`) 를 그대로 재사용하되,
ROS2·59-DOF coupled joint 는 제외하고 USD 에셋만 `asset/ALLEX/ALLEX_Hand.usd`
로 교체한다. Armature patch 는 raw USD 값을 그대로 쓰기 위해 disable 된다.
DIP/IP follower 는 USD 쪽에서 PhysicsDriveAPI 가 이미 제거되어 있어 런타임
patch 없이도 Newton equality constraint 가 유일한 구동자가 된다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import omni.usd
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from pxr import UsdGeom, Gf

from ..allex.core.joint_controller import ALLEXJointController
from ..allex.core.simulation_loop import ALLEXSimulationLoop
from .config import JOINT_CONFIG_PATH as _HYSTERESIS_JOINT_CONFIG


_EXT_ROOT = Path(__file__).resolve().parent.parent.parent
_HAND_USD_PATH = _EXT_ROOT / "asset" / "ALLEX" / "ALLEX_Hand.usd"
_HAND_PRIM_PATH = "/ALLEX_Hand"
_HAND_SPAWN_Z = 0.1
_CAMERA_EYE = [0.25, 0.15, 0.7]
_CAMERA_TARGET = [0.0, 0.0, 0.5]
_CAMERA_PRIM_PATH = "/OmniverseKit_Persp"


class HysteresisScenario:
    """Hysteresis 모드 시나리오 — 핸드 USD 하나만 stage에 올리고 TrajectoryPlayer로 구동."""

    def __init__(self):
        self._joint_controller = ALLEXJointController()
        self._simulation_loop = ALLEXSimulationLoop()
        self._articulation: SingleArticulation | None = None
        self._trajectory_player = None
        self._initialized = False
        self._seed_pose: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Setup / Load
    # ------------------------------------------------------------------
    def setup(self) -> None:
        """카메라 뷰 세팅."""
        set_camera_view(
            eye=_CAMERA_EYE,
            target=_CAMERA_TARGET,
            camera_prim_path=_CAMERA_PRIM_PATH,
        )
        print("[Hysteresis] Scenario setup complete")

    def load_example_assets(self):
        """Stage에 `ALLEX_Hand.usd` 를 레퍼런스하고 Articulation 핸들 반환."""
        if not _HAND_USD_PATH.exists():
            raise FileNotFoundError(f"ALLEX_Hand.usd not found: {_HAND_USD_PATH}")

        # Swap the equality table so Newton injects only the 5 R-hand
        # DIP/IP couplings this scene actually contains (no 7 L_*/Waist_*
        # unmatched warnings).
        from ..allex.core import newton_bridge as _nb
        _nb.set_equality_config(_HYSTERESIS_JOINT_CONFIG)

        add_reference_to_stage(str(_HAND_USD_PATH), _HAND_PRIM_PATH)

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(_HAND_PRIM_PATH)
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, _HAND_SPAWN_Z))
        xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(0.0, 0.70711, 0.0, 0.70711)
        )

        self._articulation = SingleArticulation(_HAND_PRIM_PATH)
        return self._articulation

    def delayed_initialization(self) -> bool:
        """RUN 직후 Articulation 초기화 + generator 장전."""
        if self._articulation is None:
            return False

        self._articulation.initialize()
        self._capture_seed_pose()
        self._setup_joint_control_generator()
        self._initialized = True
        print("[Hysteresis] Articulation initialized")
        return True

    def reset(self) -> None:
        self._simulation_loop.reset()
        if self._articulation is not None:
            self._capture_seed_pose()
            self._setup_joint_control_generator()
        print("[Hysteresis] Reset complete")

    def update(self, step: float):
        return self._simulation_loop.update(step)

    # ------------------------------------------------------------------
    # Internal — generator wiring
    # ------------------------------------------------------------------
    def _capture_seed_pose(self) -> None:
        pose = self._articulation.get_joint_positions()
        if hasattr(pose, "detach"):
            pose = pose.detach().cpu().numpy()
        self._seed_pose = np.asarray(pose, dtype=np.float32).reshape(-1)

    def _setup_joint_control_generator(self) -> None:
        def get_target_positions():
            if self._trajectory_player is not None and self._trajectory_player.is_active():
                traj_target = self._trajectory_player.get_current_target()
                if traj_target is not None:
                    return traj_target.tolist()
            return self._seed_pose.tolist() if self._seed_pose is not None else []

        def _traj_active():
            return self._trajectory_player is not None and self._trajectory_player.is_active()

        generator = self._joint_controller.create_joint_control_generator(
            articulation=self._articulation,
            get_target_positions_func=get_target_positions,
            is_external_active_fn=_traj_active,
        )
        self._simulation_loop.set_script_generator(generator)

    # ------------------------------------------------------------------
    # Public API (mirror of ALLEXDigitalTwin)
    # ------------------------------------------------------------------
    def get_articulation(self):
        return self._articulation

    def get_seed_pose(self) -> np.ndarray | None:
        return self._seed_pose

    def set_trajectory_player(self, player) -> None:
        if self._trajectory_player is not None:
            self._trajectory_player.stop()
        self._trajectory_player = player

    def get_trajectory_player(self):
        return self._trajectory_player

    def is_simulation_running(self) -> bool:
        return self._simulation_loop.is_running()

    def stop_simulation(self) -> None:
        self._simulation_loop.stop()

    @property
    def hand_prim_path(self) -> str:
        return _HAND_PRIM_PATH
