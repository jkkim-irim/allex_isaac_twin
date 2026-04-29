# 아키텍처 결정 기록

코드만 봐서는 *왜* 그렇게 짰는지 유추 안 되는 결정들. CLAUDE.md 가 invariant
("이건 건드리면 안 됨") 위주라면, 여기는 *대안 vs 채택 이유* 위주.

---

## 1. JSON + `utils/*_settings_utils.py` 패턴

`src/allex/config/<X>.json` (데이터) + `src/allex/utils/<X>_settings_utils.py`
(로더/적용 함수) 두 파일로 나눔.

**대안과 비교**:

| 옵션 | 장점 | 채택 안 한 이유 |
|---|---|---|
| 단일 `.py` (`_CONFIG = {...}` + apply 함수) | 코멘트 자유, 타입 풍부 (`math.pi`, `0xFF...` 리터럴) | 외부 도구가 못 읽음 |
| **JSON + utils** *(채택)* | 외부 도구·향후 GUI 가 읽기 쉬움, 데이터/로직 분리 | 코멘트 손실, hex 리터럴은 string 으로 우회 |
| YAML/TOML | 코멘트 + 타입 모두 가능 | 의존성 추가, 표준화 약함 |

**언제 분리하나**:
- 런타임에 자주 튜닝하는 파라미터 (physics, gravcomp body 리스트, 그룹 allowlist)
- 향후 외부 도구가 읽을 수도 있는 것 (ros2 토픽 매핑, UI 색상)

**언제 그냥 `.py` 로 두나**:
- 코드 안에서만 쓰이는 상수 (`utils/constants.py`)
- 한 번 정해지면 안 바뀌는 invariant

UI 헬퍼 (`UIComponentFactory`, `ButtonStyleManager`) 도 `ui_settings_utils.py`
에 함께 둠 — `contact_force_viz.py` 가 `UIComponentFactory` 를 쓰는데, `ui.py`
에 두면 `ui.py ↔ contact_force_viz.py` 순환 import 발생. utils 가 양쪽 다
import 가능한 중립 위치.

---

## 2. 단일 `ui.py` 통합 (vs 패널별 파일)

이전엔 `ui_utils/` 안에 `world_controls.py`, `ros2_controls.py`,
`traj_studio.py`, `visualizer_controls.py`, `ui_components.py`, `ui_styles.py`
+ `ui_builder.py` 따로 있었음. 모두 합쳐 단일 `src/allex/ui.py` (~810 LOC) 로
통합.

**근거**:
- 각 패널 < 250 LOC 라 별도 파일 분리 임계 미달 (참조: `dvcc/01_refactoring_guide.md`
  "선 병합, 후 분리: 800줄 이상이거나 명확한 재사용 포인트가 보일 때만 분리")
- 패널 간 cross-reference (`self._ui._scenario`, `self._ui._showcase_logger`) 가
  많아 한 파일 안에 있는 게 추적하기 쉬움
- `ui_builder.py` 는 단순 wire-up 역할이라 `AllExUI` 오케스트레이터 클래스로
  통합 — 진입점이 `from .allex.ui import AllExUI` 한 줄로 단순화됨

**예외 (utils/ 에 분리)**:
- `showcase_logger.py` (623 LOC, CSV + rosbag 처리, UI 의존 없음)
- `contact_force_viz.py` (547 LOC, MuJoCo 컨택트 데이터 처리, debug_draw)

  → 도메인 서브시스템이지 UI 패널이 아님. UI 패널은 토글 버튼만 가짐.

---

## 3. 중력보상 활성화 전략

MuJoCo Warp 의 `qfrc_gravcomp` 커널은 **빌드 타임 게이트** 가 있음:
`m.ngravcomp > 0` (즉 `(body_gravcomp > 0).any()`) 이어야 매 스텝 실행됨.
런타임에 `body_gravcomp[i] = 1.0` 만 써넣어도 `ngravcomp` 는 안 바뀌므로 무효.

**채택한 방법**: `physics_config.json::newton.gravcomp.bodies` 에 보상할 body
이름 나열 → `newton_bridge.py` 의 `_apply_gravcomp(builder)` 가 `finalize`
직전에 `builder.custom_attributes["mujoco:gravcomp"]` 에 일괄 인증.

**대안 (검토했으나 안 함)**:
- USD `mjc:gravcomp` 직접 인증: GUI 로 67 body 일일이 클릭 필요 + USD 변경
  추적이 git 에서 어려움 (binary crate)
- runtime 만 인증: ngravcomp 게이트 때문에 안 됨

**`actuatorgravcomp = false` (passive 라우팅) 선택 이유**:
- `qfrc_actuator` 에 PD only, `qfrc_gravcomp` 에 보상 only → 토크 분리 분석 가능
- `actuatorgravcomp = true` 면 `qfrc_actuator` 가 PD + 보상 합산이라 디버깅 시
  뺄셈 필요

검증 도구: `core/gravcomp_debug.py::GravcompTorqueProbe` — joint 단위로 PD /
Grav / Sum 출력 (probe 자체는 default 미사용, 진단용으로 활성화).

---

## 4. Showcase logger pose-aligned start

Trajectory player 가 1.5 s ramp + transit 후 "팔짱 자세" 같은 reference 자세에
도달하기까지의 prefix 가 CSV 에 들어가서 rosbag 과 비교하기 어려움.

**채택**: rosbag 첫 프레임의 17 joint 자세를 미리 읽어두고, 시뮬 자세가
RMS<0.05 rad 안에 들어올 때까지 CSV 행 작성 보류. 매칭되는 순간 `_step_idx=0`
리셋 후 첫 행 기록 → CSV `t=0` = rosbag `t=0` 자동 정렬.

**대안 (검토)**:
- 하드코딩 shift (`-5.36s`): trajectory 마다 달라서 매번 수동 조정 필요
- 사용자 버튼: 정확한 시점 누르기 어려움

**Side effect / 주의**:
- 30 s 안에 매칭 안 되면 timeout → 그 시점부터 기록 시작 (안전 폴백)
- `_ALIGN_ROSBAG = None` 으로 두면 옛 즉시-기록 동작으로 폴백 (다른 비교 대상
  없는 자유 trajectory 에 사용)

---

## 5. Showcase logger CSV torque = PD + gravcomp 합산

실제 로봇의 `joint_torque` 토픽은 컨트롤러 내부 중력보상이 더해진 최종 모터
명령 토크. 시뮬 측 CSV 도 동일한 reference frame 으로 맞추기 위해
`qfrc_actuator + qfrc_gravcomp` 를 한 컬럼에 기록 (분리값은 probe 로 확인).

`actuatorgravcomp = false` 이라 두 어레이가 자연스럽게 분리되어 있고, 합산만
하면 됨.

---

## 6. trajectory_generate vs 데이터 폴더 trajectory

`src/allex/trajectory/` (Python 모듈) 와 최상위 `trajectory/` (CSV 데이터
폴더) 가 같은 이름이라 `from .trajectory import ...` 와 `Path("trajectory")`
가 시각적으로 헷갈림.

**해결**: Python 모듈만 `trajectory_generate/` 로 rename. 데이터 폴더는 그대로
`trajectory/` (외부 도구·문서 호환). 모듈 import 도 `from .trajectory_generate
import TrajectoryPlayer` 로 명확.
