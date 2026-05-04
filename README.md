# ALLEX IsaacTwin

ALLEX 휴머노이드 로봇의 Real2Sim 디지털 트윈 확장(Extension). Isaac Sim 내부에서 Newton(MuJoCo Warp) 물리 엔진으로 ALLEX를 구동하며 다음 입력 경로를 지원합니다:

- **ROS2 실시간 미러링** — 실제 로봇의 `joint_positions_deg` 토픽을 200 Hz 로 추종
- **Traj Studio (CSV 궤적 재생)** — Hermite 스플라인 보간 + via 이벤트 기반 PD/torque ramp
- **Real/Sim Replay (kinematic replay)** — rosbag 에서 변환한 sim/real CSV 두 채널을 직접 키네마틱 덮어쓰기 + force/torque 시각화 오버레이로 동시 비교

부가 시각화로 **Force/Torque Visualizer** (USD prim 기반 force vector + torque ring, sim/real 채널 분리), **Contact Force Visualizer** (debug-draw immediate-mode), **Realtime Torque Plotter** (matplotlib Tk 별도 프로세스) 를 제공합니다.

## 요구사항

- NVIDIA Isaac Sim **6.0.0** (conda env 내 pip 설치, Python 3.12)
- ROS2 (Isaac Sim 내장 브리지 사용 — 별도 ROS2 설치 불필요)

> Isaac Sim 실행 전에 `source /opt/ros/*/setup.bash`를 **절대로** 실행하지 마십시오.
> 시스템 ROS2 와 내장 ROS2 브리지가 충돌하여 코어 덤프가 발생합니다.

## 설치

1. conda env (Python 3.12)에 Isaac Sim **6.0.0** 설치:

   ```bash
   conda create -n isaacsim python=3.12 -y
   conda activate isaacsim
   pip install isaacsim==6.0.0.0 --extra-index-url https://pypi.nvidia.com
   ```

2. 본 폴더를 Isaac Sim 의 `extsUser/` 하위로 복사:

   ```
   $CONDA_PREFIX/lib/python3.12/site-packages/isaacsim/extsUser/allex_isaac_twin/
   ```

3. Isaac Sim 실행 → `Window > Extensions` → `ALLEX` 검색 후 활성화

### 편의용 `cdisaac` 별칭

`conda activate isaacsim` 시 자동 등록되는 단축키로, 현재 env 의 Isaac Sim 루트로 이동합니다.

```bash
cdisaac   # → $CONDA_PREFIX/lib/python3.12/site-packages/isaacsim
```

수동 등록:

```bash
cat > "$CONDA_PREFIX/etc/conda/activate.d/aliases.sh" <<'EOF'
alias cdisaac='cd "$CONDA_PREFIX/lib/python3.12/site-packages/isaacsim"'
EOF
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/aliases.sh" <<'EOF'
unalias cdisaac 2>/dev/null
EOF
```

## 사용법

UI 는 4 + 1 패널로 구성됩니다 (`src/allex/ui.py`):

1. **World Controls** — LOAD (스테이지 + ALLEX 로드) / RESET (0 pose 3 초 ramp) / RUN (Newton 물리 기동)
2. **ROS2 Bridge** — Domain ID 지정 → Initialize → Subscriber 시작으로 실제 로봇 조인트 실시간 미러링
3. **Traj Studio** — `trajectory/<그룹>/*.csv` 그룹 선택 후 Run/Stop/Reset
   - **Raw mode (no spline)**: rosbag 에서 추출한 1 kHz 타깃 CSV 를 zero-order hold 로 그대로 재생 (Hermite 스플라인 우회)
   - **Run**: 선택한 그룹 재생 + (allowlist 등록 시) Showcase CSV 자동 기록
   - **Stop**: 현재 타깃 홀드 / **Reset**: 0 pose 로 3 초 ramp
4. **Visualizer** — 4 종 시각화 + Torque Plotter
   - **Force Visualizer** (USD prim 기반 force vector, 양손 12 link) — Viz Mode `real / sim / both` 선택
   - **Torque Rings** (관절축 둘레의 토크 링) — 동일하게 `real / sim / both` 모드
   - **Robot Opacity** — 로봇 mesh 반투명화로 시각화 prim 가시성 향상
   - **Debug draw** 체크박스: ON 시 USD prim 백엔드(`ForceTorqueVisualizer`) 가 비활성화되고 `contact_config.json` 기반 immediate-mode contact pair 가 그려지는 `ContactForceVisualizer` 로 전환 (mutually exclusive)
   - **Torque Plot (Body / Hand)**: matplotlib Tk 별도 subprocess 로 실시간 토크 플로팅 — Plot Hz / Plot Mode (`Rolling 5s` / `Cumulative`) / Save on Stop (PNG + CSV)
5. **Real/Sim Replay** — `trajectory/<그룹>/showcase_*.csv` (sim) 와 `rosbag_*.csv` (real) 두 채널 중 main source 를 선택해 키네마틱 재생, 다른 채널은 force/torque 오버레이 전용으로 사용

## 핵심 기능

### Newton(MuJoCo Warp) 물리 백엔드

PhysX 대신 `isaacsim.physics.newton` 으로 전환. 모든 물리 설정은 JSON 한 곳에서 관리.

- `src/allex/config/physics_config.json` — `NewtonConfig`, solver(MuJoCo), USD `/physicsScene`, MJCF equality tuning, **중력보상 body 리스트**, **joint friction**, **armature override (floor strategy)** 를 한 곳에서 오버라이드
- `src/allex/utils/sim_settings_utils.py` — JSON 로더 + `apply_newton_config` / `apply_gravcomp_to_builder` / `apply_usd_physics_scene` 적용 함수, `get_armature_overrides()` / `get_default_armature()` accessor
- `src/allex/core/newton_bridge.py` — `newton.ModelBuilder.finalize` 를 sentinel attribute 패턴으로 monkey-patch 해서 다음을 주입:
  - MJCF `<equality>` 제약 (Newton USD importer 가 누락)
  - `mjc:gravcomp` 인증
  - `joint_config.json` 기반 **joint friction**
  - `physics_config.json::allex.armature` 기반 **passive joint armature floor**
- Physics frequency 는 `physics_config.json::world.physics_dt` 의 역수 (기본 50 Hz, dt=0.02). UI / scenario 가 하드코딩 없이 JSON 에서 derive
- 중력 `-9.81 m/s²`, `pd_scale = π/180` 으로 SI 단위 드라이브 게인 일치화

#### Armature floor (passive joint stabilization)

Equality constraint 로 묶인 passive joint 가 측정값 그대로(매우 작은 armature)면 explicit integrator 가 발산하기 쉬워, 더 큰 `physics_dt` 에서 제어가 폭주합니다. `physics_config.json::allex.armature` 의 floor 전략으로 안정화:

```json
"armature": {
  "default_armature": 0.0,
  "armature_overrides": {
    "Waist_Upper_Pitch_Joint": 2.0,
    "Waist_Pitch_Dummy_Joint": 2.0,
    "Waist_Lower_Pitch_Joint": 0.5
  }
}
```

`newton_bridge._patch_equality_armature` 가 finalize 직전에 equality constraint 의 follower joint 를 순회하며 `joint_armature[k] = max(current, floor)` 로 갱신 — **현재값이 floor 보다 작을 때만** 끌어올리고, 큰 값은 측정값을 보존.

### MuJoCo Warp 중력보상 (gravity compensation)

USD body 67 개 전체에 `mjc:gravcomp = 1.0` 인증되어 매 스텝 `qfrc_gravcomp` 커널이 자세 의존 모멘트까지 자동 계산. `mjc:actuatorgravcomp = false` (passive 라우팅) 이라 다음 분리 가능:

| MuJoCo array (`mjw_data.*`) | 의미 |
|---|---|
| `qfrc_actuator` | PD 컨트롤러 출력만 |
| `qfrc_gravcomp` | 중력보상 토크만 |

- 활성화 게이트: `m.ngravcomp > 0` — 빌드 타임에 최소 1 body 가 nonzero 여야 커널이 도는 구조라 `physics_config.json::newton.gravcomp.bodies` 리스트로 일괄 인증
- 검증용 probe: `src/allex/core/gravcomp_debug.py::GravcompTorqueProbe` — 특정 joint 의 `PD / Grav / Sum` 분리 출력

### 실시간 PD 게인 · 토크 리밋 램프 제어

CSV 궤적에 파생 컬럼(`K_pos_*`, `K_vel_*`, `trq_lim_*`)을 실어 **via point 마다 PD 게인과 actuator 토크 리밋을 동적으로 교체**할 수 있습니다. 순간적 step 변경이 아니라 **지정 기간(기본 1.0 s) 동안 현재값→목표값으로 선형 보간**(ramp)하여, PD 불연속으로 인한 관절 스냅을 원천 차단합니다.

구현 포인트 (`src/allex/trajectory_generate/trajectory_player.py`):

- Newton `Model.joint_target_ke / joint_target_kd / joint_effort_limit` 에 **zero-copy torch view** 로 GPU 메모리를 직접 갱신 → `ArticulationView.set_dof_*` (콜당 ~140 ms) 오버헤드 완전 제거
- 매 physics step 당 1회 `solver._update_joint_dof_properties` 로 MuJoCo `gainprm/biasprm` 동기화
- CUDA graph capture 는 유지 (게인 write 는 view 기반이므로 그래프 재기록을 유발하지 않음)
- CSV 한 파일 안에서 **멀티 섹션 헤더** 지원 (중간에 `duration,...` 행이 나오면 그 시점부터 새 스키마) — PD 섹션과 토크 섹션을 자유롭게 혼용 가능

### CSV 궤적 포맷

`trajectory/<그룹>/<관절그룹>.csv` — 헤더 컬럼 순서는 자유:

```
duration, joint_1..N, trq_lim_1..N, K_pos_1..N, K_vel_1..N
```

- `duration ≤ 0` 행은 주석/섹션 마커로 무시 (예: `-1,팔짱 [9.0 sec]`)
- 첫 데이터 행의 `duration` 은 초기 hold 시간 (시간 0 에서 시작)
- 비어있는 셀은 NaN → 해당 항목은 변경하지 않음 (이전 값 유지)
- Position 은 degree → rad 변환, PD 게인 · 토크 리밋은 CSV 원본 단위 그대로

내부적으로 Hermite 스플라인으로 via point 를 보간 (`src/allex/trajectory_generate/hermite_spline.py`).

### Force/Torque Visualizer

USD prim 기반 시각화 — sim/real 두 채널을 동시에 띄울 수 있습니다 (`Both` 모드 시 동일 link 에 빨강 + 시안 두 prim 이 겹쳐 그려짐).

- `src/allex/core/force_torque_visualizer.py` — articulation attach 후 매 step 갱신
- `src/allex/config/viz_config.json` — gain / scale clip / ring offset / prim 이름 / asset 경로 등 모든 튜닝 파라미터를 외부화 (`viz_config.py` 가 module-level 상수로 expose)
- 토크 링은 group 별 gain (`gain_shoulder / elbow / wrist / finger / default`) 로 부위별 스케일링
- Force vector 는 `force_vec_thick.usda` (head/shaft 분리 mesh) 를 사용해 두께 + 길이 모두 시각적으로 의미 있게 매핑
- `set_external_torque_sources()` / `push_external_torque()` 로 외부 채널(예: replay rosbag torque) 주입 가능
- `set_replay_force_visible()` 로 kinematic replay 중 force 채널만 토글 가능

#### Contact Force Visualizer (debug-draw immediate-mode)

`src/allex/utils/contact_force_viz.py` — `omni.isaac.debug_draw` 기반의 별도 백엔드. `contact_config.json` 의 collision pair allowlist 에 정의된 페어만 매 스텝 contact normal force 를 읽어 화살표로 그립니다. Aggregate group (예: `R_Palm_Net_Force`) 은 여러 pair 의 normal-force scalar 를 합산해 `origin_shape` 의 live world transform 에 단일 화살표로 출력 — 움직이는 body 위에서도 정확히 따라옵니다.

USD prim 기반 Force Visualizer 와 **mutually exclusive** (Visualizer 패널의 `Debug draw` 체크박스로 전환).

### Realtime Torque Plotter

`src/allex/utils/torque_plotter.py` + `src/allex/utils/plot_proc.py` — matplotlib Tk window 를 **별도 subprocess** 로 띄워 메인 GIL 영향 없이 실시간 토크 플로팅.

- **Plot Mode**: `Rolling 5s` (최근 5 초만 표시) / `Cumulative` (시작부터 누적)
- **Plot Hz**: 메인 프로세스 → subprocess 로 보내는 샘플 레이트
- **Save on Stop**: 종료 시 `data/torque_plots/torque_<body|hand>_<ts>.{png,csv}` 자동 저장
- **Replay mode**: `push_replay_frame(t, sim_torque, real_torque)` 로 두 채널을 동시에 비교 (real solid line + sim dashed)
- 비단조(non-monotonic) 시간 입력은 `reset_buffer()` 로 자동 처리 (Reset / 재시작 후 t 가 0 으로 돌아가도 안전)
- Singleton 관리: extension reload 시 누수된 Tk window 가 살아남지 않도록 `register_singleton()` 이 이전 인스턴스를 stop 후 교체, `clear_singletons()` 가 shutdown 시 일괄 정리

### Feedforward Torque Manager

`src/allex/core/feedforward_torque.py` — Newton model 의 `qfrc_applied` 에 zero-copy view 로 직접 써넣어, PD 출력에 더해지는 feedforward 토크 채널을 제공. CUDA graph 친화적 (array 주소를 바꾸지 않고 contents 만 갱신).

- `apply(tau)` / `clear()` / `read_current()` / `get_last()` API
- TrajectoryPlayer 의 via 이벤트로 토크 ramp 시 사용
- Visualizer 가 `qfrc_actuator + qfrc_gravcomp + qfrc_applied` 를 합산해 토크 링/플로터에 표시

### Kinematic CSV Replay

`src/allex/replay/csv_replayer.py` + `src/allex/replay/showcase_reader.py` — CSV 의 joint position 을 직접 articulation 에 덮어써 키네마틱하게 재생 (PD/중력/충돌 응답 없음). 두 reader 를 동시에 로드 가능:

- **main reader**: 관절 키네마틱 + 한쪽 viz 채널 (sim 또는 real)
- **secondary reader** (선택): 관절은 안 건드리고 **반대 쪽 viz 채널**만 — 같은 시간축에서 sim ↔ real torque/force 를 겹쳐서 비교

스캔 규칙: `trajectory/<group>/showcase_*.csv` → sim 채널, `rosbag_*.csv` → real 채널. 양쪽이 존재하면 자동으로 두 reader 를 페어링합니다.

### Showcase 데이터 로거

Traj Studio 의 Run 시점에 `data/showcase/showcase_sim_data_<ts>.csv` 를 자동 기록. 그룹별 활성화 + rosbag 자세 정렬 지원.

- **그룹 allowlist** (`src/allex/config/showcase_logger_config.json::logged_groups`): 등록된 그룹 Run 시에만 CSV 기록. 미등록 그룹은 `[ALLEX][Showcase] group '<name>' not in logged_groups → CSV skipped` 로그 후 skip
- **Pose-aligned start**: `_ALIGN_ROSBAG` 으로 지정한 rosbag 첫 프레임 자세에 시뮬 자세가 매칭될 때까지 (RMS<0.05 rad) CSV 기록 보류 → CSV `t=0` 이 rosbag `t=0` 과 일치 (ramp-in + transit 자동 크롭)
- **Torque 컬럼 = `qfrc_actuator + qfrc_gravcomp`**: PD 토크 + 중력보상 합산 — 실제 로봇 `joint_torque` 토픽 (컨트롤러 내부 보상 포함된 모터 명령) 과 같은 reference frame 으로 비교 가능

### 설정 단일화 (JSON + utils 패턴)

런타임 튜닝 가능한 모든 설정은 `src/allex/config/*.json` 으로 통일, 적용 로직은 `src/allex/utils/*_settings_utils.py` 가 담당:

| JSON | 적용 모듈 | 내용 |
|---|---|---|
| `physics_config.json` | `sim_settings_utils.py` | NewtonConfig + MuJoCo solver + USD physicsScene + 중력보상 body + **armature override (floor)** |
| `ros2_config.json` | `ros2_settings_utils.py` | Domain ID, 토픽 매핑, 14 outbound topics, joint groups |
| `ui_config.json` | `ui_settings_utils.py` | UIColors / UILayout / UIConfig (헬퍼 클래스 포함) |
| `showcase_logger_config.json` | `showcase_logger.is_group_logged()` | CSV 기록 그룹 allowlist |
| `viz_config.json` | `viz_config.py` | Force/Torque visualizer gain·scale·prim·asset 경로 |
| `contact_config.json` | `contact_force_viz.py` | Contact force debug-draw pair allowlist + aggregate group |
| `joint_config.json` | `config/__init__.py` | `drive_gains` (SI 게인) + `ui` (각도 범위) + `coupled_joints` + `equality_constraints` + **`joint_friction`** |

`tools/regen_joint_config.py` 로 MJCF 에서 `joint_names`, `coupled_joints`, `equality_constraints` 를 자동 재생성 (`drive_gains` 와 `ui` 는 재생성 시 보존).

## ROS2 Topics

기본 Domain ID `77`, RMW `rmw_cyclonedds_cpp`. 환경변수(`ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION`) 또는 UI 에서 변경 가능.

### 조인트 위치 토픽

14 그룹 × 2 모드 = 28 토픽. 실시간 미러링 대상.

| 그룹            | 토픽                                                | 관절 수 |
| --------------- | --------------------------------------------------- | ------- |
| Right Arm       | `/robot_outbound_data/Arm_R_theOne/{mode}`          | 7       |
| Left Arm        | `/robot_outbound_data/Arm_L_theOne/{mode}`          | 7       |
| Right Hand (×5) | `/robot_outbound_data/Hand_R_{finger}_wir/{mode}`   | 3 each  |
| Left Hand (×5)  | `/robot_outbound_data/Hand_L_{finger}_wir/{mode}`   | 3 each  |
| Waist           | `/robot_outbound_data/theOne_waist/{mode}`          | 2       |
| Neck            | `/robot_outbound_data/theOne_neck/{mode}`           | 2       |

- `{mode}`: `joint_positions_deg` (current) 또는 `joint_ang_target_deg` (desired)
- `{finger}`: `thumb`, `index`, `middle`, `ring`, `little`
- 메시지 타입: `std_msgs/msg/Float64MultiArray` (degree)

## 디렉터리 구조

```
allex_isaac_twin/
├── config/extension.toml          # Extension 메타
├── asset/
│   ├── ALLEX/ALLEX.usd            # ALLEX USD 모델 (mjc:gravcomp 인증 포함)
│   ├── ALLEX/ALLEX_Hand.usd       # 손 단독 USD
│   ├── ALLEX/mjcf/ALLEX.xml       # MJCF 원본 (joint_config 재생성 기반)
│   ├── prop/                      # one_kg / two_kg / three_kg 덤벨 등
│   └── utils/                     # torque_viz.usd · force_vec.usda · force_vec_thick.usda
├── src/
│   ├── extension.py               # Isaac Sim Extension 엔트리
│   ├── global_variables.py        # EXTENSION_TITLE 등
│   ├── allex/                     # ALLEX 메인 패키지
│   │   ├── __init__.py
│   │   ├── scenario.py            # Real2Sim orchestrator (replayer/player/ff lifecycle)
│   │   ├── ui.py                  # 4 + 1 UI 패널 (World / ROS2 / Traj Studio / Visualizer / Replay)
│   │   ├── config/                # JSON 7개 + viz_config.py loader
│   │   ├── core/                  # asset_manager · joint_controller · newton_bridge · gravcomp_debug
│   │   │                          #   force_torque_visualizer · feedforward_torque · simulation_loop
│   │   ├── replay/                # CsvReplayer · ShowcaseReader (kinematic replay)
│   │   ├── ros2/                  # ROS2 노드 + 통신
│   │   ├── trajectory_generate/   # Hermite 스플라인 · TrajectoryPlayer
│   │   └── utils/                 # *_settings_utils · constants · showcase_logger
│   │                              #   contact_force_viz · torque_plotter · plot_proc · showcase_replay
│   └── hysteresis/                # Hysteresis 별도 시나리오 + 전용 윈도우
├── trajectory/                    # 궤적 데이터 (CSV 그룹) + Showcase / rosbag CSV
│   ├── demo1_dynamic_group/
│   ├── CES_260102_group/
│   └── aml_hysteresis_group/      # Hysteresis 시나리오용 (real_data + sim_measured)
├── data/                          # 런타임 출력 (showcase CSV, torque plot PNG/CSV) — gitignored
├── dvcc/                          # 프로젝트 메타 노트
├── patches/                       # Newton force access patch
└── tools/
    ├── regen_joint_config.py      # MJCF → joint_config.json 재생성
    ├── rosbag_to_csv.py           # rosbag2 (mcap) → TrajStudio CSV 변환
    └── downsample_bag.py          # rosbag2 토픽 단위 다운샘플링
```

## 로봇 스펙

- 총 관절 수: 60 (49 active + 11 coupled)
- Coupled joints: master 관절 비율로 자동 계산 (`src/allex/config/joint_config.json::coupled_joints`)
- 스폰 위치: `(0, 0, 0.685)`

## 최근 변경사항

`feature/force_torque_visualizer` 브랜치 (main 대비 누적):

- **feat(force_torque_visualizer)**: USD prim 기반 force vector + torque ring sim/real 채널 분리 시각화. `viz_config.json` 으로 gain/scale/asset 경로 외부화, `force_vec_thick.usda` 로 화살표 두께 + 길이 매핑, `set_external_torque_sources()` 로 외부 채널 주입 API
- **feat(contact_force_viz)**: `omni.isaac.debug_draw` 기반 contact pair 시각화 백엔드 추가 (`contact_config.json` allowlist + aggregate group), Force Visualizer 와 mutually exclusive UI 토글. Aggregate origin 은 Fabric API 로 runtime pose 추종
- **feat(torque_plotter)**: matplotlib Tk 별도 subprocess 로 실시간 토크 플로팅 — Rolling/Cumulative 모드, Save on Stop, replay-mode 채널 (sim solid + real dashed), 비단조 시간 안전 처리
- **feat(feedforward_torque)**: Newton `qfrc_applied` zero-copy view 로 FF 토크 채널 추가 (CUDA graph 유지)
- **feat(replay)**: kinematic CSV replay 시스템 (`src/allex/replay/`) — sim/real 두 채널 동시 로드, 한 채널은 관절 키네마틱 + 다른 채널은 viz 오버레이 전용
- **feat(physics)**: passive joint armature floor 전략 (`physics_config.json::allex.armature`) — equality constraint follower 의 armature 가 floor 미만이면 강제 상향, 더 큰 `physics_dt` 에서 발산 방지
- **feat(physics)**: `joint_config.json::joint_friction` 으로 finalize 시점 friction 주입
- **feat(traj)**: via 이벤트 기반 PD 게인 · 토크 리밋 실시간 ramp 제어 (기본 1.0 s 선형 보간), CSV 멀티 섹션 헤더 지원, zero-copy torch view 로 GPU 직접 갱신 + CUDA graph 유지. Raw mode (zero-order hold) 추가
- **feat(physics_hz)**: physics frequency 를 `physics_config.json::world.physics_dt` 에서 derive (하드코딩 제거)
- **fix(stability)**: Newton finalize 의 patch 가드를 module-level flag → 함수 sentinel attribute 로 전환해 reload 시 patch 누적 방지. Replayer/Plotter 등 stale state (lock/flag/cache/singleton) 명시 정리
- **fix(ros2)**: ros2_config 인덱스를 joint_config 와 정렬, GIL lock 완화
- **tools**: `rosbag_to_csv.py` (mcap → TrajStudio CSV), `downsample_bag.py` (토픽 단위 다운샘플링)

베이스 (main 시점):

- **refactor(structure)**: `src/<X>` → `src/allex/<X>` 패키지 재구성, `ui_utils/` + `ui_builder.py` → 단일 `src/allex/ui.py` 통합, `trajectory/` → `trajectory_generate/` rename (데이터 폴더 `trajectory/` 와 구분)
- **feat(config)**: physics / ros2 / ui / showcase_logger 설정을 JSON + `utils/*_settings_utils.py` 패턴으로 분리
- **feat(physics)**: MuJoCo Warp 중력보상 67 body 활성화 (USD `mjc:gravcomp` 인증 + `newton_bridge` finalize 후크). `qfrc_actuator` (PD only) 와 `qfrc_gravcomp` (보상 only) 분리 가능
- **feat(showcase)**: rosbag 첫 프레임 자세 매칭 후 CSV 시작 (`_ALIGN_ROSBAG`), torque 컬럼 = PD + gravcomp 합산, `showcase_logger_config.json::logged_groups` 그룹 allowlist
- **feat(hysteresis)**: 별도 ScrollingWindow 로 Hysteresis 시나리오 추가 (`src/hysteresis/`)
