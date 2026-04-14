# ALLEX IsaacTwin

ALLEX 휴머노이드 로봇의 Real2Sim 디지털 트윈 확장. 실제 로봇이 ROS2로 publish하는 관절 위치 데이터를 구독하여, Isaac Sim 내 ALLEX 로봇이 실시간으로 따라 움직인다.

## 요구사항

- NVIDIA Isaac Sim 4.5+
- ROS2 (Isaac Sim 내장 bridge 사용, 별도 설치 불필요)

## 설치

1. 이 폴더를 Isaac Sim의 `extsUser/` 디렉토리에 복사:
   ```
   <isaac-sim-root>/extsUser/ALLEX_IsaacTwin/
   ```

2. Isaac Sim 실행 후 `Window > Extensions`에서 `ALLEX` 검색 후 활성화

## 사용법

1. **LOAD** — 스테이지 생성 + ALLEX 로봇 로딩
2. **RUN** — 시뮬레이션 시작
3. **Initialize ROS2** — ROS2 노드 생성 (Domain ID: 77)
4. **Subscriber** — 28개 토픽 구독 시작 (14 current + 14 desired)
5. **Current | Desired** — 제어 모드 전환

## ROS2 토픽

Domain ID: `77` | RMW: `rmw_fastrtps_cpp`

실제 로봇이 아래 14개 그룹의 토픽을 publish하면 Isaac Sim 로봇이 따라 움직인다.

| 그룹 | 토픽 경로 | 관절 수 |
|------|----------|---------|
| 오른팔 | `/robot_outbound_data/Arm_R_theOne/{mode}` | 7 |
| 왼팔 | `/robot_outbound_data/Arm_L_theOne/{mode}` | 7 |
| 오른손 엄지 | `/robot_outbound_data/Hand_R_thumb_wir/{mode}` | 3 |
| 오른손 검지 | `/robot_outbound_data/Hand_R_index_wir/{mode}` | 3 |
| 오른손 중지 | `/robot_outbound_data/Hand_R_middle_wir/{mode}` | 3 |
| 오른손 약지 | `/robot_outbound_data/Hand_R_ring_wir/{mode}` | 3 |
| 오른손 소지 | `/robot_outbound_data/Hand_R_little_wir/{mode}` | 3 |
| 왼손 엄지 | `/robot_outbound_data/Hand_L_thumb_wir/{mode}` | 3 |
| 왼손 검지 | `/robot_outbound_data/Hand_L_index_wir/{mode}` | 3 |
| 왼손 중지 | `/robot_outbound_data/Hand_L_middle_wir/{mode}` | 3 |
| 왼손 약지 | `/robot_outbound_data/Hand_L_ring_wir/{mode}` | 3 |
| 왼손 소지 | `/robot_outbound_data/Hand_L_little_wir/{mode}` | 3 |
| 허리 | `/robot_outbound_data/theOne_waist/{mode}` | 2 |
| 목 | `/robot_outbound_data/theOne_neck/{mode}` | 2 |

- `{mode}`: `joint_positions_deg` (current) 또는 `joint_ang_target_deg` (desired)
- 메시지 타입: `std_msgs/msg/Float64MultiArray`
- 단위: degree

## 디렉토리 구조

```
ALLEX_IsaacTwin/
├── config/extension.toml       # Isaac Sim 확장 설정
├── asset/ALLEX/ALLEX.usd       # 로봇 USD 모델
├── asset/utils/                # force_viz.usd, torque_viz.usd (추후 사용)
├── src/
│   ├── extension.py            # 확장 진입점
│   ├── scenario.py             # Real2Sim 오케스트레이터
│   ├── ui_builder.py           # UI 구성 및 이벤트 처리
│   ├── config/                 # ROS2, UI, 관절 설정
│   ├── core/                   # 에셋 로딩, 관절 제어, 시뮬레이션 루프
│   ├── ros2/                   # ROS2 노드 및 통신 관리
│   └── utils/                  # 상수 정의
└── temp/                       # 보관용 코드 (visualization, sensor)
```

## 로봇 사양

- 전체 관절: 59 (49 active + 10 coupled)
- Coupled Joint: master 관절에서 비율로 자동 계산
- 스폰 위치: `(0, 0, 0.685)`
