# ALLEX IsaacTwin

ALLEX 휴머노이드 로봇의 Real2Sim 디지털 트윈 확장(Extension). Isaac Sim 내부에서 Newton(MuJoCo Warp) 물리 엔진으로 ALLEX를 구동하며, **ROS2 실시간 미러링**과 **CSV 기반 궤적 재생(Traj Studio)** 두 가지 경로를 지원합니다.

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

1. **LOAD** — 스테이지 생성 + ALLEX 로봇 로드
2. **RUN** — 시뮬레이션 시작 (Newton 물리 기동)
3. **Traj Studio** — CSV 그룹 선택 후 재생/정지/리셋
   - **Run**: 선택한 `trajectory/<그룹>/*.csv` 재생
   - **Stop**: 현재 타깃 홀드
   - **Reset**: 모든 관절을 0 pose 로 3 초에 걸쳐 부드럽게 램프
4. **ROS2** — (선택) Domain ID 지정 → Initialize → Subscriber 시작으로 실제 로봇 조인트 실시간 미러링
5. **Force Visualizer** — 양손 링크(12 prim)에 힘 벡터 시각화 토글

## 핵심 기능

### Newton(MuJoCo Warp) 물리 백엔드

PhysX 대신 `isaacsim.physics.newton` 으로 전환. 모든 물리 설정은 코드 내 구성으로 일원화되어 있습니다.

- `src/config/physics_settings.py` — `NewtonConfig`, solver(MuJoCo), USD `/physicsScene`, MJCF equality tuning 을 한 곳에서 오버라이드
- `src/core/newton_bridge.py` — MJCF `<equality>` 제약을 `newton.ModelBuilder.finalize` 후 재주입 (Newton USD importer 가 누락하는 부분 보완)
- Physics frequency **200 Hz**, 중력 `-9.81 m/s²`, `pd_scale = π/180` 으로 SI 단위 드라이브 게인 일치화

### 실시간 PD 게인 · 토크 리밋 램프 제어 (NEW)

CSV 궤적에 파생 컬럼(`K_pos_*`, `K_vel_*`, `trq_lim_*`)을 실어 **via point 마다 PD 게인과 actuator 토크 리밋을 동적으로 교체**할 수 있습니다. 순간적 step 변경이 아니라 **지정 기간(기본 1.0 s) 동안 현재값→목표값으로 선형 보간**(ramp)하여, PD 불연속으로 인한 관절 스냅을 원천 차단합니다.

구현 포인트 (`src/trajectory/trajectory_player.py`):

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

내부적으로 Hermite 스플라인으로 via point 를 보간 (`src/trajectory/hermite_spline.py`).

### 조인트 설정 단일화

기존 `src/joint_config.json` / `config/physics.toml` 을 통합하여 `src/config/joint_config.json` 하나로 관리:

- `drive_gains` — 관절별 `(stiffness, damping)` (SI 단위, MJCF 의도값)
- `ui` — `EFFECTIVE_JOINTS`, 조인트 각도 범위 등 UI 상수
- `joint_names`, `coupled_joints` — `tools/regen_joint_config.py` 로 MJCF 에서 자동 재생성 (`drive_gains` 와 `ui` 는 재생성 시 보존)

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
├── config/extension.toml           # Extension 메타
├── asset/
│   ├── ALLEX/ALLEX.usd             # ALLEX USD 모델
│   ├── ALLEX/mjcf/ALLEX.xml        # MJCF 원본 (joint_config 재생성 기반)
│   ├── prop/                       # one_kg / two_kg / three_kg 덤벨 등
│   └── utils/                      # force_viz.usd 등
├── src/
│   ├── extension.py                # Extension 엔트리
│   ├── scenario.py                 # Real2Sim orchestrator
│   ├── ui_builder.py               # UI 엔트리 (컨트롤러에 위임)
│   ├── config/                     # physics_settings · joint_config.json · UI/ROS2 설정
│   ├── core/                       # asset_manager · joint_controller · newton_bridge
│   ├── ros2/                       # ROS2 노드 + 통신
│   ├── trajectory/                 # Hermite 스플라인 · TrajectoryPlayer
│   └── ui_utils/                   # TrajStudio · VisualizerControls 등
├── trajectory/
│   ├── demo1_dynamic_group/        # 궤적 데모 세트 (CES_260102 등)
│   ├── demo2_dumbbell_group/
│   └── demo4_powerade_group/
└── tools/regen_joint_config.py
```

## 로봇 스펙

- 총 관절 수: 60 (49 active + 11 coupled)
- Coupled joints: master 관절 비율로 자동 계산 (`src/config/joint_config.json`)
- 스폰 위치: `(0, 0, 0.685)`

## 최근 변경사항

- **feat(traj)**: via 이벤트 기반 PD 게인 · 토크 리밋 실시간 ramp 제어 (기본 1.0 s 선형 보간)
- **feat(traj)**: TrajectoryPlayer 에 `hold_pose` 팩토리 + Reset 버튼 (zero pose 램프)
- **feat(traj)**: CSV 멀티 섹션 헤더 지원 (`trq_lim_*`, `K_pos_*`, `K_vel_*` 자유 혼용)
- **perf(traj)**: Newton Model 배열 zero-copy torch view 로 drive property 갱신, CUDA graph capture 유지
- **refactor(config)**: `physics.toml` 제거 → `src/config/physics_settings.py` 로 통합; `joint_config.json` 을 `src/config/` 하위로 이동 및 `drive_gains`/`ui` 섹션 병합
- **feat(assets)**: 덤벨 등 prop USD 추가, `demo1/2/4` 궤적 세트 추가
- **chore(physics)**: 중력 활성화(`-9.81`), physics 200 Hz, `pd_scale = π/180` 로 SI 드라이브 게인 정합
