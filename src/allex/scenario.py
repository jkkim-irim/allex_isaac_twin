"""ALLEX Digital Twin orchestrator — Real2Sim 관절 위치 매핑.

이 모듈이 extension 의 단일 진입점. UI 패널들(`src/allex/ui.py`)은
모두 ``ALLEXDigitalTwin`` 인스턴스 하나를 통해 core/* 서브시스템을
간접 호출한다 (UI ↛ core 직접 의존 금지 — 단방향 호출 유지).

Lifecycle:
    1. ``__init__`` — 서브시스템 인스턴스화 (asset/joint/loop), gravcomp probe 옵셔널
    2. ``setup`` — 카메라 + coupled joint config 로드 (1회)
    3. ``load_example_assets`` — USD 로드 + articulation 초기화
    4. ``update(step)`` — 매 step (gravcomp routing apply + sim loop tick)
    5. ``reset`` — 시스템 전체 재초기화 (mirror/logger 클리어)

상시 동작 서브시스템 (lazy build, 매 step 호출):
    * MotorStateMirror — motor-domain K_m/Kv_m/τ_m → joint-domain warp kernel
      + ``solver._update_joint_dof_properties`` 동기화. trajectory_player 와 독립.
    * SimStateLogger — Run 시작 시점에 활성, q/τ/K_p/K_d/τ_lim 을 CSV 로 기록.
"""

import os
from .core import ALLEXInitializer, ALLEXAssetManager, ALLEXJointController, ALLEXSimulationLoop


class ALLEXDigitalTwin:
    """ALLEX Real2Sim 디지털 트윈 orchestrator.

    UI 와 core 사이 mediator. 외부에서는 이 클래스의 public 메서드만 호출하고,
    내부 ``_articulation`` / ``_ros2_manager`` / ``_trajectory_player`` 등은
    UI 가 ``getattr`` 로 접근 — direct setattr 는 lifecycle 깨짐."""

    def __init__(self):
        self._initializer = ALLEXInitializer()
        self._asset_manager = ALLEXAssetManager()
        self._joint_controller = ALLEXJointController()
        self._simulation_loop = ALLEXSimulationLoop()
        self._articulation = None
        self._ros2_manager = None
        self._trajectory_player = None
        self._gravcomp_probe = self._build_gravcomp_probe()
        # actuator_gravcomp routing 은 finalize 후 mjw_model 이 준비돼야 적용 가능 →
        # 매 step 호출하다 첫 성공 시 True 로 막음 (CUDA graph 재기록 방지).
        self._actuator_gravcomp_applied = False
        # MotorStateMirror — lazy build (Newton stage ready 후 첫 step 시점에 생성).
        # 매 physics step 모든 active 모터의 K_j(q) / D_j(q) / τ_j_max(q) 를
        # nominal_motor_gains + 현재 q 로 변환해 model.joint_target_ke/kd/effort_limit
        # 에 직접 write. trajectory_player 와 독립 — Run/Idle/Reset 모두에서 동작.
        self._motor_mirror = None
        self._motor_mirror_failed = False
        # SimStateLogger — lazy build, None until start_sim_state_log() 호출.
        self._sim_state_logger = None

    def setup(self): 
        """시나리오 초기 설정 — 카메라 뷰 + coupled joint config 로드"""
        self._initializer.setup_camera_view()

        extension_root = os.path.dirname(os.path.abspath(__file__))
        joint_config_path = os.path.join(extension_root, "config", "joint_config.json")
        self._joint_controller.load_coupled_joint_config(joint_config_path)

        print("ALLEX Digital Twin setup complete")

    def load_example_assets(self):
        """로봇 USD 에셋 로딩 및 제너레이터 설정"""
        self._articulation = self._asset_manager.load_robot_asset()

        if self._articulation is None:
            return None

        init_success = self._asset_manager.initialize_articulation()
        if init_success:
            self._setup_joint_control_generator()
        else:
            print("RUN 버튼을 누르면 Articulation이 초기화됩니다")

        return self._articulation

    def delayed_initialization(self):
        """지연 초기화 — 시뮬레이션 시작 후 Articulation 초기화"""
        if self._articulation is None:
            return False

        init_success = self._asset_manager.initialize_articulation()
        if not init_success:
            return False

        self._initializer.initialize_joint_positions(self._articulation)
        self._setup_joint_control_generator()
        print("Articulation 초기화 완료")
        return True

    def reset(self):
        """시스템 리셋"""
        self._initializer.reset(self._articulation)
        self._simulation_loop.reset()
        # mjw_model 이 새로 만들어지므로 routing 재적용 필요
        self._actuator_gravcomp_applied = False
        # Drop stale mirror — Newton model is rebuilt; new mirror will be
        # constructed by _motor_mirror_step on the next physics tick and
        # re-wired into any active TrajectoryPlayer.
        self._motor_mirror = None
        self._motor_mirror_failed = False

        if self._articulation is not None:
            self._setup_joint_control_generator()

        print("시스템 리셋 완료")

    def update(self, step: float):
        """매 physics step마다 호출"""
        # actuator_gravcomp routing 1회성 적용 (첫 step 에서 mjw_model 준비되면 성공)
        if not self._actuator_gravcomp_applied:
            try:
                from .utils.sim_settings_utils import apply_actuator_gravcomp_runtime
                if apply_actuator_gravcomp_runtime():
                    self._actuator_gravcomp_applied = True
            except Exception as exc:
                print(f"[ALLEX][Gravcomp] runtime apply error: {exc}")
                self._actuator_gravcomp_applied = True  # 재시도 안 함

        if self._gravcomp_probe is not None and self._articulation is not None:
            try:
                self._gravcomp_probe.step(self._articulation)
            except Exception as exc:
                print(f"[ALLEX][Gravcomp] probe.step error: {exc}")
                self._gravcomp_probe = None  # 한 번 실패하면 비활성
        return self._simulation_loop.update(step)

    @staticmethod
    def _build_gravcomp_probe():
        """`physics_config.json::debug.gravcomp_probe` 설정으로 probe 생성. 비활성/오류면 None."""
        try:
            from .utils.sim_settings_utils import _load
            cfg = _load().get("debug", {}).get("gravcomp_probe", {}) or {}
            if not bool(cfg.get("enabled", False)):
                return None
            from .core.gravcomp_debug import GravcompTorqueProbe
            joint = str(cfg.get("joint_name", "R_Shoulder_Pitch_Joint"))
            period = int(cfg.get("period", 200))
            print(f"[ALLEX][Gravcomp] probe enabled: joint={joint}, period={period}")
            return GravcompTorqueProbe(joint_name=joint, period=period)
        except Exception as exc:
            print(f"[ALLEX][Gravcomp] probe init failed: {exc}")
            return None

    # ========================================
    # 내부 헬퍼
    # ========================================
    def _setup_joint_control_generator(self):
        """관절 제어 제너레이터 생성 및 시뮬레이션 루프에 설정"""
        self._initializer.initialize_joint_positions(self._articulation)

        def get_target_positions():
            # Trajectory playback takes priority when active.
            if self._trajectory_player is not None and self._trajectory_player.is_active():
                traj_target = self._trajectory_player.get_current_target()
                if traj_target is not None:
                    return traj_target.tolist()

            ros2_positions = self._joint_controller.get_unified_target_positions()
            if ros2_positions and any(pos != 0.0 for pos in ros2_positions):
                return ros2_positions
            return self._initializer.target_joint_positions

        def _traj_active():
            return self._trajectory_player is not None and self._trajectory_player.is_active()

        def get_target_velocities():
            """Velocity target paired with the most recent position target.

            Trajectory mode → analytic dense_vel row from Hermite. Other modes
            (ROS2 mirror / idle hold) carry no velocity reference, so return
            None so the controller falls back to position-only.
            """
            player = self._trajectory_player
            if player is None or not player.is_active():
                return None
            vel = player.get_current_velocity_target()
            if vel is None:
                return None
            return vel.tolist()

        generator = self._joint_controller.create_joint_control_generator(
            articulation=self._articulation,
            get_target_positions_func=get_target_positions,
            is_external_active_fn=_traj_active,
            pre_step_fn=self._motor_mirror_step,
            get_target_velocities_func=get_target_velocities,
        )
        self._simulation_loop.set_script_generator(generator)

    def start_sim_state_log(self, log_every: int = 1, start_offset_steps: int = 0) -> None:
        """SimStateLogger 시작 (lazy build — Newton stage ready 후 첫 호출에 생성).

        ``start_offset_steps`` > 0 이면 처음 N step 캡처를 skip — 외부 데이터
        (rosbag 등) 의 t=0 과 시뮬레이터 t=N/physics_hz 자세를 정렬할 때 사용.
        """
        if self._sim_state_logger is None:
            # 아직 mirror 가 없으면 build 실패 → 다음 step 에서 mirror_step 이 build 하면 재시도
            if self._motor_mirror is None:
                print("[ALLEX][SimLog] motor_mirror not ready yet; logger will start after mirror init")
            # logger 는 motor_mirror 와 같은 Newton model/solver 를 공유
            # motor_mirror 가 없어도 Newton stage 가 있으면 독립 build 가능
            try:
                from isaacsim.physics.newton import acquire_stage  # type: ignore[import-not-found]
                stage = acquire_stage()
                if stage is None:
                    print("[ALLEX][SimLog] Newton stage not ready; retry after RUN")
                    return
                model = getattr(stage, "model", None)
                solver = getattr(stage, "solver", None)
                if model is None or solver is None:
                    print("[ALLEX][SimLog] model/solver not ready yet")
                    return
                from .trajectory_generate.sim_state_logger import SimStateLogger
                from .utils.sim_settings_utils import get_physics_hz
                physics_hz = get_physics_hz()
                self._sim_state_logger = SimStateLogger(
                    self._articulation, model, solver, log_every=log_every
                )
            except Exception as exc:
                print(f"[ALLEX][SimLog] build failed: {exc}")
                return
        from .utils.sim_settings_utils import get_physics_hz
        self._sim_state_logger.start(
            log_every=log_every,
            physics_hz=get_physics_hz(),
            start_offset_steps=start_offset_steps,
        )

    def stop_sim_state_log(self) -> str:
        """SimStateLogger 중단 후 CSV 저장. 출력 경로 반환."""
        if self._sim_state_logger is None or not self._sim_state_logger.is_active():
            print("[ALLEX][SimLog] logger not active")
            return ""
        return self._sim_state_logger.stop()

    def _motor_mirror_step(self) -> None:
        """매 physics step pre-step hook: lazy 빌드 + update() 호출.

        Newton stage 가 ready 되기 전 호출되면 silent skip. 빌드 실패 시
        `_motor_mirror_failed=True` 로 영구 비활성화 (재시도 안 함).
        """
        if self._motor_mirror_failed:
            return
        if self._motor_mirror is None:
            try:
                from isaacsim.physics.newton import acquire_stage  # type: ignore[import-not-found]
                stage = acquire_stage()
                if stage is None:
                    return  # not ready yet, retry next step
                model = getattr(stage, "model", None)
                solver = getattr(stage, "solver", None)
                if model is None or solver is None:
                    return
                from .trajectory_generate.motor_state_mirror import MotorStateMirror
                self._motor_mirror = MotorStateMirror(self._articulation, model, solver, stage)
            except Exception as exc:
                print(f"[ALLEX][Mirror] init failed: {exc}; disabling motor_mirror")
                self._motor_mirror_failed = True
                return
            # Late-bind to any TrajectoryPlayer that arrived before the mirror.
            if (self._trajectory_player is not None
                    and getattr(self._trajectory_player, "_motor_mirror", None) is None):
                self._trajectory_player.set_motor_mirror(self._motor_mirror)
        try:
            self._motor_mirror.update()
        except Exception as exc:
            print(f"[ALLEX][Mirror] update failed: {exc}; disabling motor_mirror")
            self._motor_mirror_failed = True

        # SimStateLogger — motor_mirror update 직후 (게인이 이미 write된 상태) 캡처
        if self._sim_state_logger is not None:
            self._sim_state_logger.step()
            # trajectory 재생이 끝나면 자동 저장
            if (self._sim_state_logger.is_active()
                    and self._trajectory_player is not None
                    and not self._trajectory_player.is_active()):
                out = self._sim_state_logger.stop()
                print(f"[ALLEX][SimLog] trajectory done → auto-saved to {out}")

    # ========================================
    # Public API
    # ========================================
    def set_ros2_manager(self, ros2_manager):
        self._ros2_manager = ros2_manager

    def set_trajectory_player(self, player):
        """Install a TrajectoryPlayer; replaces any existing one."""
        if self._trajectory_player is not None:
            try:
                self._trajectory_player.stop()
            except Exception:
                pass
        self._trajectory_player = player
        # Wire mirror immediately if it exists; otherwise _motor_mirror_step
        # will late-bind when it builds the mirror.
        if (player is not None and self._motor_mirror is not None
                and hasattr(player, "set_motor_mirror")):
            player.set_motor_mirror(self._motor_mirror)

    def get_trajectory_player(self):
        return self._trajectory_player

    def get_robot_info(self):
        return self._asset_manager.get_joint_info()

    def is_simulation_running(self):
        return self._simulation_loop.is_running()

    def stop_simulation(self):
        self._simulation_loop.stop()
