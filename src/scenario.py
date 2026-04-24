"""
ALLEX Digital Twin 메인 시나리오 — Real2Sim 관절 위치 매핑
"""

import logging
import os
from .core import (
    ALLEXInitializer,
    ALLEXAssetManager,
    ALLEXJointController,
    ALLEXSimulationLoop,
    ForceTorqueVisualizer,
    RuntimeGainFFController,
    FeedforwardTorqueManager,
)
from .ui_utils.torque_plotter import TorquePlotter, register_singleton


logger = logging.getLogger("allex.scenario")


class ALLEXDigitalTwin:
    """ALLEX Real2Sim 디지털 트윈 — ROS2 관절 데이터로 시뮬레이션 로봇 구동"""

    def __init__(self):
        self._initializer = ALLEXInitializer()
        self._asset_manager = ALLEXAssetManager()
        self._joint_controller = ALLEXJointController()
        self._simulation_loop = ALLEXSimulationLoop()
        self._articulation = None
        self._ros2_manager = None
        self._trajectory_player = None
        self._visualizer = None
        self._gain_ff_controller = None
        self._gain_ff_reader = None
        # Fallback step counter used when the currently active trajectory
        # player has no sample index (e.g. ROS2-driven target stream).
        self._gain_ff_step_counter = 0

        # General-purpose feedforward torque manager (control.joint_f 쓰기 +
        # 최근 값 보관). Newton 이 qfrc_applied 를 expose 하지 않아 쓴 값
        # 되읽는 공식 API 가 없어 이 매니저가 사실상 ground truth 역할.
        self._ff_manager = FeedforwardTorqueManager()

        # Torque plotters (body = arm+waist+neck, hand = fingers).
        # Dormant until .start() is called from the UI or python console.
        self._torque_plotter_body: TorquePlotter | None = None
        self._torque_plotter_hand: TorquePlotter | None = None

    def setup(self):
        """시나리오 초기 설정 — 카메라 뷰 + coupled joint config 로드"""
        self._initializer.setup_camera_view()

        extension_root = os.path.dirname(os.path.abspath(__file__))
        joint_config_path = os.path.join(extension_root, "joint_config.json")
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

        self._initializer.reset(self._articulation)
        self._simulation_loop.reset()

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
        done = self._simulation_loop.update(step)
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

    # 호환용 shim — 예전 코드 경로에서 호출되는 경우 대비
    def _setup_visualizer(self):
        if self._visualizer is None:
            self._create_visualizer_prims()
        if self._articulation is not None:
            self._attach_visualizer_articulation()

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

        # Build runtime gain/FF controller + reader against the currently
        # installed trajectory (if any). Whether apply_step is called is
        # decided every step based on reader readiness.
        self._setup_runtime_gain_ff()

        def _pull_gain_ff():
            ctrl = self._gain_ff_controller
            reader = self._gain_ff_reader
            if ctrl is None or reader is None or not reader.is_ready():
                return
            # Prefer the trajectory player's sample counter so gain_ff.csv
            # stays phase-locked with trajectory CSVs at the same hz.
            sample_idx = None
            tp = self._trajectory_player
            if tp is not None and tp.is_active():
                sample_idx = getattr(tp, "_sample_idx", None)
            if sample_idx is None:
                sample_idx = self._gain_ff_step_counter
                self._gain_ff_step_counter += 1
            sample = reader.get_sample(int(sample_idx))
            ctrl.apply_step(sample)

        # Build / rebuild torque plotters against the current articulation
        # so they pick up the fresh dof_names list. Plotters stay dormant
        # until start() is called (UI button or python console).
        self._setup_torque_plotters()

        # Per-step dt for plotter time axis — falls back to UIConfig PHYSICS_DT.
        try:
            from .config.ui_config import UIConfig as _UICfg
            _plot_dt = float(_UICfg.PHYSICS_DT)
        except Exception:
            _plot_dt = 1.0 / 200.0

        def _push_torque_sample():
            body = self._torque_plotter_body
            hand = self._torque_plotter_hand
            if body is not None and body.is_running():
                body.apply_step(_plot_dt)
            if hand is not None and hand.is_running():
                hand.apply_step(_plot_dt)

        generator = self._joint_controller.create_joint_control_generator(
            articulation=self._articulation,
            get_target_positions_func=get_target_positions,
            is_external_active_fn=_traj_active,
            runtime_gain_ff_fn=_pull_gain_ff,
            torque_plot_fn=_push_torque_sample,
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

    def _setup_torque_plotters(self):
        """(Re)create TorquePlotter instances for body + hand subsets.

        If plotters were already running when articulation is reinitialized
        (e.g. after Stop/Play), we tear them down first so the new instance
        binds to the fresh articulation view.
        """
        if self._articulation is None:
            return

        ff_provider = None
        if self._gain_ff_controller is not None:
            ff_provider = self._gain_ff_controller.get_last_applied_ff

        # Stop any previous plotter subprocesses cleanly before rebuilding.
        self._stop_torque_plotters()

        try:
            self._torque_plotter_body = TorquePlotter(
                articulation=self._articulation,
                ff_provider=ff_provider,
                subset="body",
                ff_manager=self._ff_manager,
            )
            register_singleton("body", self._torque_plotter_body)
        except Exception as exc:
            logger.warning(f"TorquePlotter(body) init failed: {exc}")
            self._torque_plotter_body = None

        try:
            self._torque_plotter_hand = TorquePlotter(
                articulation=self._articulation,
                ff_provider=ff_provider,
                subset="hand",
                ff_manager=self._ff_manager,
            )
            register_singleton("hand", self._torque_plotter_hand)
        except Exception as exc:
            logger.warning(f"TorquePlotter(hand) init failed: {exc}")
            self._torque_plotter_hand = None

    def _setup_runtime_gain_ff(self):
        """Instantiate RuntimeGainFFController + GainFFCsvReader.

        The controller is always built when the articulation is ready. Whether
        it actually modifies gains / injects torque at runtime is decided
        per-step by ``reader.is_ready()`` — if the trajectory's csv_dir has
        no gain_ff.csv (or it has no relevant columns), apply_step is skipped.
        """
        self._gain_ff_step_counter = 0
        self._gain_ff_reader = None

        if self._articulation is None:
            self._gain_ff_controller = None
            return

        try:
            dof_names = list(self._articulation.dof_names or [])
        except Exception:
            dof_names = []
        if not dof_names:
            print("[ALLEX][GainFF] articulation has no dof_names yet; skipping setup")
            self._gain_ff_controller = None
            return

        try:
            self._gain_ff_controller = RuntimeGainFFController(
                articulation=self._articulation,
                dof_names=dof_names,
            )
        except Exception as exc:
            print(f"[ALLEX][GainFF] controller init failed: {exc}")
            self._gain_ff_controller = None
            return

        # CSV is driven by the currently installed trajectory player's csv_dir.
        tp = self._trajectory_player
        csv_dir = getattr(tp, "_csv_dir", None) if tp is not None else None
        if csv_dir is None:
            # No active trajectory yet; reader stays None. apply_step becomes
            # a no-op until set_trajectory_player rebuilds the reader.
            return
        from pathlib import Path
        csv_path = Path(csv_dir) / "gain_ff.csv"
        try:
            from .trajectory import GainFFCsvReader
            self._gain_ff_reader = GainFFCsvReader(
                csv_path=csv_path,
                dof_names=dof_names,
            )
        except Exception as exc:
            print(f"[ALLEX][GainFF] reader init failed: {exc}")
            self._gain_ff_reader = None

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
        # Re-bind gain/FF reader against the new trajectory's csv_dir so that
        # gain_ff.csv shipped alongside the trajectory takes effect immediately.
        if self._articulation is not None:
            self._setup_runtime_gain_ff()

    def get_trajectory_player(self):
        return self._trajectory_player

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
