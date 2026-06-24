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

import logging
import os
from .core import (
    ALLEXInitializer,
    ALLEXAssetManager,
    ALLEXJointController,
    ALLEXSimulationLoop,
    ForceTorqueVisualizer,
    FeedforwardTorqueManager,
)
from .utils.torque_plotter import DataPlotter, register_singleton
from .config.viz_config import (
    TORQUE_PLOT_WINDOW_SECONDS,
    TORQUE_PLOT_Y_LIM_BODY,
    TORQUE_PLOT_Y_LIM_HAND,
)


logger = logging.getLogger("allex.scenario")


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
        self._csv_replayer = None
        self._visualizer = None
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

        # General-purpose feedforward torque manager (control.joint_f 쓰기 +
        # 최근 값 보관). Newton 이 qfrc_applied 를 expose 하지 않아 쓴 값
        # 되읽는 공식 API 가 없어 이 매니저가 사실상 ground truth 역할.
        self._ff_manager = FeedforwardTorqueManager()

        # Torque plotters (body = arm+waist+neck, hand = fingers).
        # Dormant until .start() is called from the UI or python console.
        self._torque_plotter_body: DataPlotter | None = None
        self._torque_plotter_hand: DataPlotter | None = None

    def setup(self):
        """시나리오 초기 설정 — 카메라 뷰 + coupled joint config 로드"""
        self._initializer.setup_camera_view()

        extension_root = os.path.dirname(os.path.abspath(__file__))
        joint_config_path = os.path.join(extension_root, "config", "joint_config.json")
        self._joint_controller.load_coupled_joint_config(joint_config_path)

        print("ALLEX Digital Twin setup complete")

    def load_example_assets(self):
        """로봇 USD 에셋 로딩 및 제너레이터 설정.

        visualizer prim 은 asset load 직후(= articulation 초기화 전) 바로 생성한다.
        Play 전에도 torque ring / force arrow prim 이 stage 에 존재해야 visibility 토글이
        의미가 있다.
        """
        self._articulation = self._asset_manager.load_robot_asset()

        if self._articulation is None:
            return None

        # 1단계: asset load 직후 — stage 기반 prim 즉시 생성 (articulation 불필요)
        self._create_visualizer_prims()

        init_success = self._asset_manager.initialize_articulation()
        if init_success:
            self._setup_joint_control_generator()
            # 2단계: articulation 초기화 완료 → LUT 만 attach
            self._attach_visualizer_articulation()
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
        # 혹시 prim 이 아직 없으면 생성, 그리고 articulation attach
        if self._visualizer is None:
            self._create_visualizer_prims()
        self._attach_visualizer_articulation()
        print("Articulation 초기화 완료")
        return True

    def reset(self):
        """시스템 리셋 — prim 은 유지, articulation 재초기화 시점에 재-attach 만 수행."""
        # 기존 plotter subprocess 는 articulation 재바인딩 전에 반드시 정리.
        self._stop_torque_plotters()

        # CSV replayer / trajectory player / FF manager 도 articulation 재바인딩 전에
        # 정리해야 stale subscription, external_torque_sources lock, 잔존 FF 토크가
        # 다음 articulation 인스턴스로 새지 않음.
        if self._csv_replayer is not None:
            try:
                self._csv_replayer.stop()
            except Exception as exc:
                logger.debug(f"reset csv_replayer stop warn: {exc}")
            self._csv_replayer = None
        if self._trajectory_player is not None:
            try:
                self._trajectory_player.stop()
            except Exception as exc:
                logger.debug(f"reset trajectory_player stop warn: {exc}")
            self._trajectory_player = None
        try:
            if self._ff_manager is not None:
                self._ff_manager.clear()
        except Exception as exc:
            logger.debug(f"reset ff_manager clear warn: {exc}")

        self._initializer.reset(self._articulation)
        self._simulation_loop.reset()
        # mjw_model 이 새로 만들어지므로 routing 재적용 필요
        self._actuator_gravcomp_applied = False
        # Drop stale mirror — Newton model is rebuilt; new mirror will be
        # constructed by _motor_mirror_step on the next physics tick and
        # re-wired into any active TrajectoryPlayer.
        self._motor_mirror = None
        self._motor_mirror_failed = False

        # visualizer prim 은 cleanup 하지 않는다. articulation 만 detach 후 재 attach.
        if self._visualizer is not None:
            try:
                self._visualizer.detach_articulation()
            except Exception as e:
                logger.warning(f"visualizer detach warn: {e}")

        if self._articulation is not None:
            self._setup_joint_control_generator()
            # prim 이 없으면 안전하게 재생성
            if self._visualizer is None:
                self._create_visualizer_prims()
            self._attach_visualizer_articulation()

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

        done = self._simulation_loop.update(step)
        # csv_replayer 는 기본적으로 omni.kit.app update stream (rendering tick) 으로
        # 자체 진행한다 (kinematic replay 는 physics 와 무관). subscription 등록 실패 시
        # fallback 으로 여기서 호출.
        if self._csv_replayer is not None and not self._csv_replayer.is_self_ticking():
            try:
                self._csv_replayer.advance()
            except Exception as e:
                logger.warning(f"csv_replayer advance warn: {e}")
        if self._visualizer is not None:
            try:
                self._visualizer.update(step)
            except Exception as e:
                logger.warning(f"visualizer update warn: {e}")
        return done

    # ========================================
    # Visualizer 접근자 (UI 에서 사용)
    # ========================================
    def get_visualizer(self):
        return self._visualizer

    def get_ff_manager(self) -> "FeedforwardTorqueManager":
        """General-purpose FF torque manager (control.joint_f 주입 + 보관)."""
        return self._ff_manager

    def _create_visualizer_prims(self):
        """1단계: articulation 없이 stage 기반 prim 만 생성."""
        if self._visualizer is not None:
            return
        try:
            self._visualizer = ForceTorqueVisualizer(stage=None)
            # 생성자 안에서 self._ensure_prims 가 호출됨 — 여기서는 확인만.
            self._visualizer.ensure_initialized()
            print("ForceTorqueVisualizer prims created (pre-articulation)")
        except Exception as e:
            logger.warning(f"ForceTorqueVisualizer prim create failed: {e}")
            self._visualizer = None

    def _attach_visualizer_articulation(self):
        """2단계: articulation 준비 완료 후 LUT 만 구축."""
        if self._visualizer is None:
            # 1단계를 놓친 경우 방어적으로 생성
            self._create_visualizer_prims()
            if self._visualizer is None:
                return
        if self._articulation is None:
            return
        try:
            self._visualizer.attach_articulation(
                self._articulation, self._joint_controller,
                ff_manager=self._ff_manager,
            )
            print("ForceTorqueVisualizer articulation attached")
        except Exception as e:
            logger.warning(f"ForceTorqueVisualizer attach failed: {e}")

        # FF manager 버퍼를 articulation dof 수에 맞춤.
        try:
            ndof = int(getattr(self._articulation, "num_dof", 0) or 0)
            if ndof > 0:
                self._ff_manager.set_num_dof(ndof)
        except Exception as e:
            logger.warning(f"FeedforwardTorqueManager resize failed: {e}")

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

        # physics_config.json::world.physics_hz 을 single source of truth 로
        # 사용. plotter 의 decim ratio (physics_hz / plot_hz) 와 apply_step 의
        # 시간축 dt 둘 다에 같은 값 derive.
        from .utils.sim_settings_utils import get_physics_hz
        physics_hz = get_physics_hz()
        physics_dt = 1.0 / physics_hz

        # Build / rebuild torque plotters against the current articulation
        # so they pick up the fresh dof_names list. Plotters stay dormant
        # until start() is called (UI button or python console).
        self._setup_torque_plotters(physics_hz)

        def _push_torque_sample():
            body = self._torque_plotter_body
            hand = self._torque_plotter_hand
            if body is not None and body.is_running():
                body.apply_step(physics_dt)
            if hand is not None and hand.is_running():
                hand.apply_step(physics_dt)

        def get_target_velocities():
            """Velocity target paired with the most recent position target.

            Trajectory mode → analytic dense_vel row from Hermite. Other modes
            (ROS2 mirror / idle hold) carry no velocity reference, so return
            None and downstream consumers fall back to zero (= hold pose).
            """
            player = self._trajectory_player
            if player is None or not player.is_active():
                return None
            vel = player.get_current_velocity_target()
            if vel is None:
                return None
            return vel.tolist()

        # 저장 — MotorStateMirror 빌드 시점에 callback 으로 전달.
        # ArticulationAction 으로는 보내지 않음 (Isaac Sim 6 의 articulation_controller
        # 가 joint_velocities CUDA tensor 에 np.isnan 직접 호출 → TypeError).
        self._get_target_velocities_fn = get_target_velocities

        generator = self._joint_controller.create_joint_control_generator(
            articulation=self._articulation,
            get_target_positions_func=get_target_positions,
            is_external_active_fn=_traj_active,
            torque_plot_fn=_push_torque_sample,
            pre_step_fn=self._motor_mirror_step,
        )
        self._simulation_loop.set_script_generator(generator)

    def _stop_torque_plotters(self):
        """Tear down any running plotter subprocesses.

        Safe to call even when plotters were never created. Used on reset
        and on extension shutdown so we never leave orphan Python Tk
        windows behind.
        """
        for attr in ("_torque_plotter_body", "_torque_plotter_hand"):
            old = getattr(self, attr, None)
            if old is None:
                continue
            try:
                old.stop()
            except Exception as exc:
                logger.debug(f"torque plotter stop warn: {exc}")
            setattr(self, attr, None)

    def _setup_torque_plotters(self, physics_hz: float):
        """(Re)create DataPlotter instances for body + hand subsets.

        If plotters were already running when articulation is reinitialized
        (e.g. after Stop/Play), we tear them down first so the new instance
        binds to the fresh articulation view.

        physics_hz: derived from physics_config.json::world.physics_dt by the
        caller. Required by DataPlotter to compute its decim ratio.
        """
        if self._articulation is None:
            return

        # Plotter 는 FF torque snapshot 을 FeedforwardTorqueManager 에서 조회.
        ff_provider = self._ff_manager.get_last
        # ROS2-fed real torque snapshot. JointController 가 없거나 메서드가
        # 없으면 None 으로 두고 plotter 는 real 라인을 비활성화한다.
        real_provider = None
        jc = self._joint_controller
        if jc is not None and hasattr(jc, "get_torque_snapshot"):
            real_provider = jc.get_torque_snapshot

        # Stop any previous plotter subprocesses cleanly before rebuilding.
        self._stop_torque_plotters()

        try:
            self._torque_plotter_body = DataPlotter(
                articulation=self._articulation,
                physics_hz=physics_hz,
                ff_provider=ff_provider,
                subset="body",
                window_s=TORQUE_PLOT_WINDOW_SECONDS,
                ff_manager=self._ff_manager,
                real_provider=real_provider,
                y_lim=TORQUE_PLOT_Y_LIM_BODY,
            )
            register_singleton("body", self._torque_plotter_body)
        except Exception as exc:
            logger.warning(f"DataPlotter(body) init failed: {exc}")
            self._torque_plotter_body = None

        try:
            self._torque_plotter_hand = DataPlotter(
                articulation=self._articulation,
                physics_hz=physics_hz,
                ff_provider=ff_provider,
                subset="hand",
                window_s=TORQUE_PLOT_WINDOW_SECONDS,
                ff_manager=self._ff_manager,
                real_provider=real_provider,
                y_lim=TORQUE_PLOT_Y_LIM_HAND,
            )
            register_singleton("hand", self._torque_plotter_hand)
        except Exception as exc:
            logger.warning(f"DataPlotter(hand) init failed: {exc}")
            self._torque_plotter_hand = None

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
                control = getattr(stage, "control", None)
                self._sim_state_logger = SimStateLogger(
                    self._articulation, model, solver, log_every=log_every,
                    control=control,
                )
                # Bind scenario so logger can read live _motor_mirror each step
                # (mirror is lazy-built on first physics step — may be None now).
                self._sim_state_logger._scenario_ref = self
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
                control = getattr(stage, "control", None)
                if control is None:
                    return  # not ready yet; retry next step
                self._motor_mirror = MotorStateMirror(
                    self._articulation, model, solver, stage, control=control,
                    get_target_vel_fn=getattr(self, "_get_target_velocities_fn", None),
                )
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
        """Install a TrajectoryPlayer; replaces any existing one.

        CsvReplayer 와 상호배타 — trajectory player 를 활성화하면 csv_replayer 가
        자동 stop.
        """
        if self._trajectory_player is not None:
            try:
                self._trajectory_player.stop()
            except Exception:
                pass
        # csv_replayer 와 동시 활성화 금지 (kinematic vs PD 충돌).
        if player is not None and self._csv_replayer is not None:
            try:
                self._csv_replayer.stop()
            except Exception:
                pass
            self._csv_replayer = None
        # 이전 CsvReplayer 가 plotter 를 replay 모드로 두고 freeze 시켜놨을 수 있음 —
        # TrajStudio 는 articulation 폴링이 필요하므로 명시적으로 해제.
        if player is not None:
            for plotter in (self._torque_plotter_body, self._torque_plotter_hand):
                if plotter is not None and hasattr(plotter, "set_replay_mode"):
                    try:
                        plotter.set_replay_mode(False)
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

    def set_csv_replayer(self, replayer):
        """Install a CsvReplayer; replaces any existing one.

        TrajectoryPlayer 와 상호배타 — set 시 trajectory player 를 자동 stop.
        replayer=None 으로 호출하면 현재 replayer 를 stop + 제거.
        """
        if self._csv_replayer is not None and self._csv_replayer is not replayer:
            try:
                self._csv_replayer.stop()
            except Exception:
                pass
        if replayer is not None and self._trajectory_player is not None:
            try:
                self._trajectory_player.stop()
            except Exception:
                pass
            self._trajectory_player = None
        self._csv_replayer = replayer

    def get_csv_replayer(self):
        return self._csv_replayer

    # ========================================
    # Torque plotter accessors (UI uses these)
    # ========================================
    def get_torque_plotter(self, subset: str = "body"):
        if subset == "hand":
            return self._torque_plotter_hand
        return self._torque_plotter_body

    def get_robot_info(self):
        return self._asset_manager.get_joint_info()

    def is_simulation_running(self):
        return self._simulation_loop.is_running()

    def stop_simulation(self):
        self._simulation_loop.stop()

    def shutdown(self):
        """Extension reload / window close hook.

        Ensures all plotter subprocesses are terminated so we do not leave
        orphan Tk windows around after the Kit extension is unloaded.
        """
        self._stop_torque_plotters()

        # Replayer / player 도 stop — extension reload 시 subscription 누수 방지.
        if self._csv_replayer is not None:
            try:
                self._csv_replayer.stop()
            except Exception as exc:
                logger.debug(f"shutdown csv_replayer stop warn: {exc}")
            self._csv_replayer = None
        if self._trajectory_player is not None:
            try:
                self._trajectory_player.stop()
            except Exception as exc:
                logger.debug(f"shutdown trajectory_player stop warn: {exc}")
            self._trajectory_player = None

        # 모듈 레벨 singleton 등록부도 비워서 reload 후 stale plotter 참조 제거.
        try:
            from .utils.torque_plotter import clear_singletons
            clear_singletons()
        except Exception:
            pass
