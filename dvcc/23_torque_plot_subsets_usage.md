# Data Plot subset / subplot 설정 가이드

`viz_config.json::torque_plot` 안에서 **어떤 joint 를 어떤 subplot 으로, 어떤 채널 (torque / position) 로 그릴지** 직접 정의할 수 있다. UI 의 "Data Plot (Body)" / "Data Plot (Hand)" 버튼이 누르는 subset 이름과 매핑된다.

> 키 이름이 `torque_plot` 인 건 historical (`DataPlotter` 로 일반화 전 이름). 데이터 의미는 channel 별로 결정됨.

---

## 1. 키 구조

```jsonc
"torque_plot": {
  "window_seconds": 20.0,            // rolling 모드 visible time window
  "y_lim_body": null,                // null → autoscale, [ymin, ymax] → 고정
  "y_lim_hand": [-2.0, 2.0],

  "subsets": {
    "_doc": "...",                   // 주석용. 코드 무시.

    "<subset_name>": [               // "body" | "hand" | (사용자 정의)
      {
        "name": "<subplot 제목>",     // 화면에 보일 라벨
        "joints": ["L_Elbow_Joint", ...],  // USD dof_name 그대로
        "y_lim": [-30, 30]                  // optional — 이 subplot 만 고정 y-axis
      },
      ...
    ]
  }
}
```

### 키별 의미

| 키 | 타입 | 설명 |
|---|---|---|
| `torque_plot.window_seconds` | float | rolling 모드의 가시 시간 윈도우 (초). cumulative 모드에선 무시. |
| `torque_plot.y_lim_body` | `null` \| `[float, float]` | body subset 의 y 축 **subset 기본** 범위. `null` 이면 autoscale. `ymin >= ymax` 면 autoscale fallback. |
| `torque_plot.y_lim_hand` | `null` \| `[float, float]` | hand subset 의 y 축 subset 기본 범위. |
| `torque_plot.subsets` | dict | subset → group 리스트. 없거나 비면 regex 기반 fallback 동작 (아래 §4). |
| `torque_plot.subsets.<subset>` | list of group | 각 entry = 한 subplot. 순서대로 위→아래 배치. |
| `<group>.name` | string | subplot 제목 (legend 위쪽). |
| `<group>.joints` | list of string | 그 subplot 에 함께 표시할 dof_name (USD 기준). |
| `<group>.channel` | optional `"torque"` \| `"pos"` (default `"torque"`) | 이 subplot 이 그릴 채널. `"torque"` 는 N m, `"pos"` 는 deg 로 표시 (내부 rad → plot_proc 에서 변환). 잘못된 값은 `"torque"` 로 fallback. |
| `<group>.y_lim` | optional `[float, float]` | 이 subplot 만의 고정 y 축 (channel 의 단위로 해석: torque=Nm, pos=deg). 우선순위: **group.y_lim > (torque channel 일 때) subset 의 y_lim_\<subset\> > autoscale**. pos channel 은 subset-level fallback 없음 — autoscale 또는 group.y_lim 지정만. |
| `<group>.highlights` | optional list of `{t0, t1, color?, alpha?, edgecolor?, linewidth?}` | 이 subplot 에 사전 등록된 시간 구간 강조 띠 (matplotlib `axvspan`). 자세한 형식은 §F. |

### Joint 이름 형식

`dof_name` 은 articulation 의 USD 표기와 1:1 매칭. 예:

- Body: `L_Shoulder_Pitch_Joint`, `R_Elbow_Joint`, `L_Wrist_Roll_Joint`, `Waist_Yaw_Joint`, `Neck_Pitch_Joint` ...
- Hand: `L_Thumb_MCP_Joint`, `R_Index_DIP_Joint`, `L_Middle_PIP_Joint` ...

전체 목록은 articulation load 후 `articulation.dof_names` 또는 `joint_config.json::active_joints` 에서 확인.

---

## 2. 동작 규칙

- **dof_names 에 없는 joint** 는 silently drop. 그 joint 만 빠지고, 나머지는 정상 표시.
- **빈 group** (drop 후 joints 가 0 개) 은 subplot 자체가 생성되지 않음.
- **빈 subset** (group 이 모두 빈 경우) 은 plot 시작 자체가 거부됨 ("no joints matched subset=..." 로그).
- `name` / `joints` 누락된 entry, 또는 `joints` 가 list 가 아닌 entry 는 silently drop.
- `subsets` 자체가 dict 가 아니면 (`null`, list, string ...) 전체 무시하고 regex fallback.

---

## 3. 예시

### A. 4개 body joint 만 (기본 제공)

```jsonc
"subsets": {
  "body": [
    {"name": "Left Shoulder Pitch",  "joints": ["L_Shoulder_Pitch_Joint"]},
    {"name": "Left Elbow",           "joints": ["L_Elbow_Joint"]},
    {"name": "Right Shoulder Pitch", "joints": ["R_Shoulder_Pitch_Joint"]},
    {"name": "Right Elbow",          "joints": ["R_Elbow_Joint"]}
  ]
}
```

→ 4개 subplot 세로로 나열, 각 subplot 에 단일 joint 의 Real / Sim 두 라인.

### B. 한 subplot 에 여러 joint (legend 그룹화)

```jsonc
"subsets": {
  "body": [
    {"name": "L Arm", "joints": [
      "L_Shoulder_Pitch_Joint", "L_Shoulder_Roll_Joint", "L_Shoulder_Yaw_Joint",
      "L_Elbow_Joint",
      "L_Wrist_Yaw_Joint", "L_Wrist_Roll_Joint", "L_Wrist_Pitch_Joint"
    ]}
  ]
}
```

→ subplot 1개에 7개 joint 의 Real/Sim 총 14 라인. Per-joint 색은 같은 hue 안에서 lightness 로 자동 분배 (real=red 계열, sim=blue 계열).

### C. Hand 의 일부만 (legend 폭주 방지)

전체 손 (20 joint × 2 = 40 entry) 은 legend 가 거대해지므로, **궁금한 손가락만** 골라 표시:

```jsonc
"subsets": {
  "hand": [
    {"name": "L Thumb", "joints": [
      "L_Thumb_CMC_Joint", "L_Thumb_MCP_Joint", "L_Thumb_IP_Joint"
    ]},
    {"name": "L Index", "joints": [
      "L_Index_MCP_Joint", "L_Index_PIP_Joint", "L_Index_DIP_Joint"
    ]}
  ]
}
```

### D. Subplot 별로 다른 y 축 범위

```jsonc
"y_lim_body": [-30, 30],   // 기본 범위
"subsets": {
  "body": [
    {"name": "L Shoulder Pitch", "joints": ["L_Shoulder_Pitch_Joint"]},          // y_lim_body 사용 → [-30, 30]
    {"name": "L Elbow",          "joints": ["L_Elbow_Joint"], "y_lim": [-80, 80]} // 이 subplot 만 [-80, 80]
  ]
}
```

→ Shoulder subplot 은 `y_lim_body` 의 [-30, 30], Elbow subplot 만 [-80, 80] 고정.

### E. Subset 자체 제거 → regex fallback

`subsets` 키를 통째로 지우거나 `"subsets": {}` 으로 두면, 코드는 regex 기반 자동 분할로 폴백:

- `hand` → `Hand L` (모든 `L_<finger>_*_Joint`) + `Hand R`
- `body` → `Arm L` + `Arm R` + `Waist + Neck` 버킷

### F. 시간 구간 강조 (axvspan)

특정 시간 윈도우를 반투명 띠로 강조 (예: trigger 활성 구간 / 외력 인가 구간 marking):

```jsonc
{
  "name": "L Elbow",
  "joints": ["L_Elbow_Joint"],
  "highlights": [
    {"t0": 10.0, "t1": 20.0, "color": "#ff8c00", "alpha": 0.15},
    {"t0": 30.0, "t1": 35.0, "color": "#dc143c", "alpha": 0.20}
  ]
}
```

| 필드 | 필수 | 기본값 | 설명 |
|---|---|---|---|
| `t0`, `t1` | ✅ | — | 시작 / 끝 sim_time (초, 절대 기준). `t1 ≤ t0` 면 silently drop. |
| `color` | ❌ | `#ff8c00` | facecolor (matplotlib hex 또는 이름). |
| `alpha` | ❌ | `0.15` | 0~1, 범위 밖이면 clamp. |
| `edgecolor` | ❌ | (없음, edge 안 그림) | 가장자리 색. |
| `linewidth` | ❌ | `0.0` (edgecolor 없을 때) / `0.8` (있을 때) | edge 두께. |

시간 기준은 plot 의 x 축과 동일한 절대 sim_time. rolling window 모드에선 `t0 ~ t1` 이 view 밖이면 자연스럽게 안 보이고, view 안에 들어오면 스크롤되며 나타남.

**색 추천 (다크 배경)**: amber `#ff8c00` (일반), crimson `#dc143c` (위험/이벤트), yellow `#ffd700` (정보), magenta `#ff00ff` (보조). 라인 color 인 red / cyan 과는 충돌하므로 피할 것.

### G. Channel mix (position + torque 한 subset 에 혼합)

같은 subset 안에서 pos channel subplot 과 torque channel subplot 을 섞어 배치 가능. 한 subplot 은 한 channel.

```jsonc
"subsets": {
  "body": [
    {"name": "L Shoulder Pitch pos",    "joints": ["L_Shoulder_Pitch_Joint"], "channel": "pos",    "y_lim": [-90, 90]},
    {"name": "L Elbow pos",             "joints": ["L_Elbow_Joint"],          "channel": "pos",    "y_lim": [-150, 30]},
    {"name": "L Shoulder Pitch torque", "joints": ["L_Shoulder_Pitch_Joint"], "channel": "torque", "y_lim": [-45, 28]},
    {"name": "L Elbow torque",          "joints": ["L_Elbow_Joint"],          "channel": "torque", "y_lim": [-45, 28]}
  ]
}
```

→ 4 개 subplot 이 vertical 로 배치, 위 2 개는 deg 단위 position, 아래 2 개는 N m 단위 torque.

**단위**: pos channel 의 데이터는 항상 **rad** 로 수집되어 (live articulation `get_joint_positions` / CSV `position_*` 컬럼) plotter 에서 **표시 직전 deg 로 변환**되어 그려진다. `y_lim` 은 표시 단위 기준으로 작성 (pos → deg, torque → Nm).

**Live mode 동작**:
- torque channel: 기존과 동일 (qfrc_actuator + joint_f → Real/Sim 라인).
- pos channel: sim 라인만 (articulation 폴링). real 라인은 ROS2 position provider 가 없어 항상 비어 있음 (실선만 그려짐).

**CSV replay mode**: 두 channel 모두 sim / real 라인이 정상 채워짐 (CSV 의 `position_*` / `torque_*` 컬럼 사용).

---

## 4. Showcase 별 swap

`viz_config_*.json` (bolt / coke / dynamic / pen / pringles ...) 마다 다른 subset 정의 가능. Showcase 교체 시 해당 파일을 `viz_config.json` 으로 복사/심볼릭링크.

기본적으로 모든 변형이 동일한 subsets (B-style: body=4 single-joint, hand=Hand L/R 전체) 로 시드돼 있다. 각 showcase 의 관심사에 맞게 자유롭게 가지치기.

---

## 5. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| plot 이 빈 화면 | 모든 group 의 joint 가 dof_names 에 없음. joint 이름 오타 확인 (`articulation.dof_names` 출력). |
| 일부 joint 만 안 보임 | 그 joint 만 dof_names 에 없음 — 표시되는 joint 들은 정상. 로그에 직접적인 경고는 없음. |
| legend 가 너무 큼 | group 당 joint 수를 줄이거나 group 을 분리. 한 subplot 의 joint 수 ≤ 5~6 권장. |
| subplot 이 너무 짧음 | `plot_proc.py::figsize` 의 row 당 inch 를 늘리거나, group 수를 줄임. |
| y 축이 의도와 다른 범위 | `y_lim_<subset>` 확인. `null` 이면 autoscale, 잘못된 형식 (예: `[2, -2]`) 도 autoscale fallback. |

---

## 6. 관련 코드

- 데이터 정의: `src/allex/config/viz_config.json::torque_plot.subsets`
- 로더 + 검증: `src/allex/config/viz_config.py::_parse_subsets` → 모듈 상수 `TORQUE_PLOT_SUBSETS`
- 사용 지점: `src/allex/utils/torque_plotter.py::DataPlotter._select_joints` + `_build_groups`
- 색상 / lightness 자동 분배: `src/allex/utils/plot_proc.py::_shade`
