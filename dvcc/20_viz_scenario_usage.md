# Real/Sim Replay — Viz Scenario Config 사용법

`Real/Sim Replay` 패널의 force vector / torque ring 시각화를 시간 기반
시나리오로 제어하는 통합 annotation 파일.

## 파일 위치

각 group 디렉토리 안에 한 개:

```
trajectory/<group>/
├── real_*.csv                    ← rosbag_to_csv.py 가 생성, 수정 X
├── sim_*.csv                     ← 있어도 되고 없어도 됨
└── viz_scenario_config.json      ← (선택) 손으로 작성
```

**파일이 없으면 기존 default 동작**: force/torque 는 default visible (mode-only + threshold-hide),
torque ring 은 user 토글.

## JSON 스키마 (전부 optional)

```json
{
  "_notes": "Real/Sim Replay 시나리오 — 시간 단위 [s] (rosbag 첫 sample 0 기준 상대시간)",

  "force_triggers": {
    "real.arm_r":                 [[33.0, 38.5]],
    "real.hand_l_thumb":          [[33.0, 35.5]],
    "sim.L_Elbow__R_Palm_Back":   [[13.5, 17.6]],
    "sim.R_Palm_Net_Force":       [[33.0, 36.8]]
  },

  "force_invert": [
    "sim.L_Elbow__R_Palm_Back"
  ],

  "force_gain": {
    "sim.L_Elbow__R_Palm_Back": 2.0,
    "real.arm_r":               0.5
  },

  "force_origin": {
    "real.arm_r":        "sim.L_Elbow__R_Palm_Back",
    "real.hand_r_index": ["link:R_Index_Distal_Link", [0.0, 0.0, -0.02]]
  },

  "ext_torque_triggers": {
    "real.arm_r":        [[33.0, 38.5]],
    "real.hand_l_thumb": "off"
  },

  "torque_ring_triggers": {
    "real": "off",
    "sim":  [[10.0, 50.0]]
  },

  "ext_joint_torque_triggers": {
    "ext_joint_torque_hand_l_index_abad":    [[0.0, 4.0]],
    "ext_joint_torque_arm_r_wrist_pitch":    [[5.0, 8.0]]
  }
}
```

`_` 로 시작하는 키 (예 `_notes`) 는 무시 (메모용).

### 공통 value 규칙

모든 trigger section (`force_triggers`, `ext_torque_triggers`,
`torque_ring_triggers`, `ext_joint_torque_triggers`) 의 value 는 다음 형식 지원:

| value 형식 | 의미 |
|---|---|
| `[[t_on, t_off], ...]` | 정상 trigger window (반드시 `t_on < t_off`) |
| `"off"` / `false` / `[]` | **explicit OFF** — 해당 key 는 replay 동안 항상 비활성 (hide / substitute 안 함) |
| key 자체 없음 / section `{}` | trigger 미설정 → 그 section 의 **default 동작** (아래 표) |

| Section | section `{}` (default) | per-key `"off"` |
|---|---|---|
| `force_triggers` | default visible (source 별 독립, 아래 §1 참조) | 그 channel 만 hide |
| `ext_torque_triggers` | default visible (small \|τ\| 는 hide_below 로 auto hide) | 그 topic 만 hide |
| `torque_ring_triggers` | mode 만 보고 default visible | 그 region 만 hide (exclusive 시멘틱 cascade) |
| `ext_joint_torque_triggers` | substitution 안 함 (internal torque 그대로) | 그 joint 는 절대 substitute 안 함 |

---

## 1) `force_triggers` — Force vector 시간 게이팅 (real ext_force + sim pair/aggregate)

### Key 형식

dotted form 두 종류:

- `"real.<topic_id>"` : real CSV 의 ext_force topic (예: `real.arm_r`,
  `real.hand_l_thumb`). showcase_reader 의 `topic_force_vec` key 와 직접 매칭.
- `"sim.<channel>"` : sim CSV 의 pair_force_vec sanitized key 또는 aggregate key
  (예: `sim.L_Elbow__R_Palm_Back`, `sim.R_Palm_Net_Force`).

> Backward-compat: `"ext_force_<id>"` legacy form 도 받지만 `"real.<id>"` 로 해석되며
> section 단위로 WARNING 1회 emit. 신규 작성은 dotted form 권장.

#### Real channel 도출

real CSV header 의 `ext_force_<id>_x/y/z` 컬럼명에서 `ext_force_` prefix 떼고
`real.` 붙이면 됨. `tools/rosbag_to_csv.py::EXT_FORCE_TOPICS` 기준 등록된 14 종:

| JSON key                  | CSV 컬럼                          | 설명                       |
|---------------------------|-----------------------------------|----------------------------|
| `real.arm_l`              | `ext_force_arm_l_{x,y,z}`         | L_Palm 외력                |
| `real.arm_r`              | `ext_force_arm_r_{x,y,z}`         | R_Palm 외력                |
| `real.shoulder_l`         | `ext_force_shoulder_l_{x,y,z}`    | L Shoulder reaction       |
| `real.shoulder_r`         | `ext_force_shoulder_r_{x,y,z}`    | R Shoulder reaction       |
| `real.hand_l_thumb`       | `ext_force_hand_l_thumb_{x,y,z}`  | L 엄지 외력                |
| `real.hand_l_index`       | `ext_force_hand_l_index_{x,y,z}`  | L 검지 외력                |
| `real.hand_l_middle`      | `ext_force_hand_l_middle_{x,y,z}` | L 중지 외력                |
| `real.hand_l_ring`        | `ext_force_hand_l_ring_{x,y,z}`   | L 약지 외력                |
| `real.hand_l_little`      | `ext_force_hand_l_little_{x,y,z}` | L 새끼 외력                |
| `real.hand_r_thumb`       | `ext_force_hand_r_thumb_{x,y,z}`  | R 엄지 외력                |
| `real.hand_r_index`       | `ext_force_hand_r_index_{x,y,z}`  | R 검지 외력                |
| `real.hand_r_middle`      | `ext_force_hand_r_middle_{x,y,z}` | R 중지 외력                |
| `real.hand_r_ring`        | `ext_force_hand_r_ring_{x,y,z}`   | R 약지 외력                |
| `real.hand_r_little`      | `ext_force_hand_r_little_{x,y,z}` | R 새끼 외력                |

#### Sim channel 도출

채널의 canonical source 는 `src/allex/config/contact_config.json`. 두 family:

- **Pair** (`contact_config.json::pairs[].name`) — `"L_Elbow <-> R_Palm_Back"` 형식.
  CSV 컬럼 `force_vec_<A> <-> <B>_{x,y,z}` 로 logged 되고, showcase_reader 의
  `sanitize_pair_name` 이 `" <-> "` → `"__"` 으로 변환해 dict key 생성.
  **JSON 키는 sanitized 형 그대로 사용** (공백/`<->` 직접 입력 X): `sim.L_Elbow__R_Palm_Back`.
- **Aggregate** (`contact_config.json::aggregate_groups[].name`) — tag 가 이미
  공백/`<->` 없는 단일 토큰이라 변환 없이 그대로: `sim.R_Palm_Net_Force`.

#### 등록된 sim 채널 (현재 contact_config.json 기준, 총 3)

| JSON key                       | CSV 컬럼 prefix                         | source                                     |
|--------------------------------|-----------------------------------------|--------------------------------------------|
| `sim.L_Elbow__R_Palm_Back`     | `force_vec_L_Elbow <-> R_Palm_Back_*`   | `pairs[0]` (왼팔 → 오른손등)               |
| `sim.L_Palm_Back__R_Elbow`     | `force_vec_L_Palm_Back <-> R_Elbow_*`   | `pairs[1]` (왼손등 ← 오른팔)               |
| `sim.R_Palm_Net_Force`         | `force_aggregate_R_Palm_Net_Force`      | `aggregate_groups[0]` (양손 clasp 합산)    |

> 신규 sim 채널 추가 흐름: `contact_config.json` 에 pair/aggregate 추가 → showcase
> logger 가 자동으로 새 컬럼 emit → JSON 에 `sim.<sanitized_name>` 키 박기. 컬럼명 →
> JSON 키 변환은 pair 만 sanitize 필요, aggregate 는 그대로.

`showcase_sim_data_20260427` 시나리오 예시 (3 motion):

| sim channel                 | active window       | 의미                            |
|-----------------------------|---------------------|---------------------------------|
| `sim.L_Elbow__R_Palm_Back`  | `[[13.5, 17.6]]`    | Motion 1: 왼팔↔오른손등 pair    |
| `sim.L_Palm_Back__R_Elbow`  | `[[20.6, 24.3]]`    | Motion 2: 오른팔↔왼손등 pair    |
| `sim.R_Palm_Net_Force`      | `[[33.0, 36.8]]`    | Motion 3: 양손 clasp aggregate  |

#### 신규 topic 추가

`tools/rosbag_to_csv.py::EXT_FORCE_TOPICS` 에 새 topic 추가 후 CSV 재생성하면, JSON 에
`real.<topic_id>` 키만 박으면 자동 visualize. 하드코딩 routing dict 없음 —
sim 쪽과 동시에 보이고 싶다면 두 source 모두에 각자 trigger entry 추가.

### 우선순위 (source 별 독립, channel 별 독립)

force_triggers 의 게이트는 **`source` (real/sim) 별로 따로** 평가. 한쪽 source 의
trigger 가 다른 source 의 게이트에 영향 주지 않음.

각 source 안에서:

1. **이 channel 의 trigger key 있음** → 시간 범위로 on/off
2. **그 source 에 다른 key 만 있고 이 channel 은 미정의** → **exclusive hide**
3. **그 source 에 key 가 통째로 없음** → default visible
   (mode-only, threshold-hide 가 작은 force 알아서 숨김)

| 이 channel key | 같은 source 의 다른 key | 결과 게이트          |
|----------------|--------------------------|----------------------|
| ✅             | 무관                     | trigger 시간 범위     |
| ❌             | ✅                       | 항상 off (exclusive) |
| ❌             | ❌                       | **default visible**   |

> 이전 버전에 있던 `SIM_TO_REAL_FORCE_ROUTE` 하드코딩 routing (sim 채널 `|F|>1mN` 이면
> 매핑된 real topic 자동 visible) 은 제거됐다. 그 의도는 이제 JSON 에 sim/real 양쪽
> entry 를 적어서 표현한다 (`showcase_sim_data_20260427` 예시 참조).

### Visualization 파이프라인

```
# real
vector  = ext_force_<topic_id>_{x,y,z}     ← real CSV (chest_origin frame)
origin  = contact_pos_<topic_id>_{x,y,z}   ← real CSV (chest_origin frame)
        → chest→world transform (sim Chest_Origin_Link world matrix, Fabric API)
prim    = /World/AllexForceViz/real/<topic_id>

# sim
vector  = force_vec_<A> <-> <B>_{x,y,z}    ← sim CSV (world frame, 변환 없음)
          또는 -normal × mag (aggregate; 손등 외향)
origin  = contact_pos / aggregate origin   ← sim CSV
prim    = /World/AllexForceViz/sim/<channel>
```

real `contact_pos_<topic_id>_*` 컬럼이 없으면 chest_origin link world 위치로 fallback.

### `force_invert` — 채널별 force vector 방향 반전

CSV 값을 음수로 push 해서 화살표를 반대 방향으로 그리는 viz-only 옵션. 데이터는
건드리지 않고 시각화만 바뀐다 (origin 위치는 그대로). 활성 시간창 내에서만 영향.

```json
"force_invert": [
  "sim.L_Elbow__R_Palm_Back",
  "sim.L_Palm_Back__R_Elbow",
  "real.arm_r"
]
```

- 항목 형식: `"sim.<channel>"` 또는 `"real.<topic_id>"` — `force_triggers` 와 같은 dotted key.
- sim aggregate (예 `sim.R_Palm_Net_Force`) 는 default 가 이미 `-normal*mag` 라
  `force_invert` 에 넣으면 부호가 한 번 더 뒤집혀 **CSV normal 방향 그대로** 출력.
- 잘못된 entry (prefix 가 sim/real 아님, dot 없음 등) 는 WARNING + skip.

### `force_gain` — 채널별 magnitude 배율

화면상 화살표 길이만 채널별로 키우거나 줄이고 싶을 때. 데이터 값은 그대로,
시각화 magnitude 만 `gain` 배. visualizer 의 global `FORCE_GAIN` (viz_config) 위에
**추가로** 곱해진다.

```json
"force_gain": {
  "sim.L_Elbow__R_Palm_Back": 2.0,
  "real.arm_r":               0.5
}
```

- 값은 **양의 finite float** — 음수/0/NaN/Inf 은 WARNING + skip (방향 반전은 `force_invert`).
- 누락된 채널은 1.0 (변화 없음).
- `force_invert` 와 같이 쓰면 `vec → -vec * gain` (둘 다 적용).
- 활성 시간창 (`force_triggers`) 안에서만 영향.

### `force_origin` — Real force vector 의 world origin 을 다른 channel/link 에 anchor

기본적으로 `real.<topic_id>` force vector 는 자기 토픽의 `contact_pos` 를 chest→world
변환해서 origin 으로 쓴다 (`topic_contact_pos` 가 CSV 에 없으면 chest 위치로 fallback).
**해당 origin 이 실제 contact 지점과 안 맞을 때** sim 채널의 contact 또는 임의 link
의 world 위치에 origin 을 맞춘다. 매 frame dynamic — link/sim 채널이 움직이면 따라감.

```json
"force_origin": {
  "real.arm_r":         "sim.L_Elbow__R_Palm_Back",
  "real.hand_r_thumb":  "link:R_Thumb_Distal_Link",
  "real.hand_r_index":  ["link:R_Index_Distal_Link", [0.0, 0.0, -0.02]]
}
```

- **Key**: `real.<topic_id>` 만. sim 채널의 origin 은 CSV 가 결정하므로 override
  불가.
- **Value** — string 또는 `[string, [x, y, z]]` 배열:
  - `"sim.<channel>"` — 그 sim 채널의 매 frame contact origin 따라감. sim push 가
    real push 보다 먼저 호출되므로 같은 frame 값 보장. 활성 sim 채널이 없으면
    fallback (CSV 기본 origin).
  - `"link:<usd_link_name>"` — `/ALLEX/<usd_link_name>` 의 world transform.
    Offset 없으면 **link 의 원점 = joint pivot** (보통 segment 의 base 쪽).
    절대 경로가 필요하면 `"link:/some/abs/Path"`.
  - `[string, [x, y, z]]` — 위 string 형식 + **link-local frame** offset (m).
    link 의 worldMatrix 로 회전·평행이동돼서 anchor 가 link 표면/끝점 등 임의
    위치로 옮겨짐. **link 의 origin 이 finger 의 시작 (joint pivot) 이므로**
    손가락 끝/중앙을 찍으려면 거의 항상 offset 이 필요. (`sim.` alias 와
    같이 쓰면 rotation 정보가 없어 offset 무시 + warning.)
- 잘못된 entry (key 가 real 아님, value 가 위 형식 아님 등) 는 WARNING + skip.
- 같은 topic 에 대해 inline alias (legacy: `force_triggers` 값 리스트에 `"sim.X"`
  문자열 섞기) 와 `force_origin` 양쪽이 정의돼 있으면 **`force_origin` 우선** +
  WARNING. 신규 코드는 `force_origin` 만 사용 권장.
- link prim 이 stage 에 없거나 fabric worldMatrix 가 없으면 1회 WARNING 후
  default origin fallback. (오타 디버깅 단서.)
- Offset 좌표계는 **link local** (link 의 joint frame). USD Composer 또는
  `asset/ALLEX/mjcf/ALLEX.xml` 의 `body pos`/child mesh transform 을 참고해서
  대략값 잡고 한 번 돌려보면서 미세조정 추천.

흔한 link 이름 예시 (`asset/ALLEX/mjcf/ALLEX.xml` 참조):
`R_Index_Distal_Link`, `R_Thumb_Distal_Link`, `R_Middle_Distal_Link`,
`L_Index_Distal_Link`, `Chest_Origin_Link`, `R_Palm_Link`, `L_Palm_Link` 등.

---

## 2) `ext_torque_triggers` — Wrench torque **ring** (EE contact_pos) 시간 게이팅

`/result/<group>/ee/ext_force` 토픽의 **Wrench.torque** 성분 (Nm) 을 **free-floating
torque ring** 으로 표시. arrow 가 아니다 — joint 의 torque ring 과 같은 asset
(`torque_viz.usd`) 을 contact_pos 위치에 띄워서 ring 의 +Z 축이 torque 벡터를
가리키게 회전시킨다.

기존 joint-attached torque ring 과는 **별개 prim namespace** (`/World/AllexTorqueRing/...`)
이라 둘이 동시에 보일 수 있음.

### Key 형식

dotted form: `"real.<topic_id>"`. real CSV header 의 `ext_torque_<id>_x/y/z` 컬럼
prefix 에서 `ext_torque_` 떼고 `real.` 붙임. topic_id 는 `force_triggers` 의 real
쪽과 동일 — `EXT_FORCE_TOPICS` 가 force/torque 토픽 매핑을 공유한다.

> Backward-compat: `"ext_torque_<id>"` 도 받지만 `"real.<id>"` 로 해석되며 section
> 단위로 WARNING 1회.
>
> Sim source 는 미지원 — Wrench torque 데이터가 sim CSV 에 없다. `"sim.*"` 키가
> 들어오면 WARNING 후 skip.

#### 등록된 모든 key (총 14, force 와 동일)

| JSON key                | CSV 컬럼                            |
|-------------------------|-------------------------------------|
| `real.arm_l`            | `ext_torque_arm_l_{x,y,z}`          |
| `real.arm_r`            | `ext_torque_arm_r_{x,y,z}`          |
| `real.shoulder_l`       | `ext_torque_shoulder_l_{x,y,z}`     |
| `real.shoulder_r`       | `ext_torque_shoulder_r_{x,y,z}`     |
| `real.hand_l_thumb` ~ `_little`  | (L 5종)                    |
| `real.hand_r_thumb` ~ `_little`  | (R 5종)                    |

> Wrench (force+torque) 가 아닌 Vector3 토픽이면 rosbag_to_csv 가 torque 자리에 0
> 으로 패딩 → 그 topic 은 항상 hide_below 로 자동 숨김.

### 우선순위

`force_triggers` 와 **독립** 게이팅 (별개 entry, 별개 prim). 같은 시점에 force arrow
와 torque ring 이 서로 다른 visibility 일 수 있음.

| `real.<topic>` key      | real 의 다른 key | 결과 게이트         |
|-------------------------|------------------|---------------------|
| `[[t_on, t_off]]`       | 무관             | 그 window 만 visible |
| `"off"` / `[]` / `false`| 무관             | 항상 hide            |
| key 없음                | 다른 key 있음    | 항상 hide (exclusive) |
| key 없음                | 다른 key 없음    | default visible (`\|τ\| < hide_below` 면 자동 hide) |

### Visualization 파이프라인

```
torque_vec   = ext_torque_<topic_id>_{x,y,z}   ← real CSV (chest_origin frame)
position     = contact_pos_<topic_id>_{x,y,z}  ← real CSV (chest_origin frame)
             → chest→world transform (sim Chest_Origin_Link world matrix, Fabric API)
ring +Z axis = torque_vec_world / |torque_vec_world|
ring scale   = clip(|τ|*EXT_TORQUE_GAIN, EXT_TORQUE_MIN_SCALE, EXT_TORQUE_MAX_SCALE)
prim_path    = /World/AllexTorqueRing/real/torque_<topic_id>
```

contact_pos 컬럼이 없으면 chest_origin link world 위치로 fallback (force arrow 와 동일).

### Visibility

- UI 의 **torque mode** (off/real/sim/both) 가 source 허용해야 visible (force arrow 가 아닌
  torque ring 이라 force_mode 가 아닌 **torque_mode** 검사)
- `|τ| < hide_below` (Nm) → ring 전체 invisible
- trigger window 밖 → invisible (exclusive 시멘틱)

### viz_config.json 연동

`viz_config.json::ext_torque` 섹션:
- `gain`: 스케일 multiplier (default 1.0)
- `min_scale` / `max_scale`: ring 크기 clamp (default 0.05 / 3.0)
- `hide_below`: |τ| 임계 [Nm] (default 0.05)
- `color_real` / `color_sim`: ring 색. null 이면 colors.real / colors.sim 사용.

---

## 3) `torque_ring_triggers` — Torque ring 시간/영역 게이팅

### Key 형식

```json
"torque_ring_triggers": {
  "<source>":           [[t_on, t_off], ...],   // 그 source 의 모든 ring
  "<source>.<region>":  [[t_on, t_off], ...]    // 그 source 의 region 만
}
```

- `<source>` ∈ `"real"` / `"sim"`
- `<region>` 은 아래 14개 중 하나. 다른 키는 WARNING + skip.

#### 모든 region key (각 source 별 동일하게 사용 가능)

##### Arm (3-tier: 전체 / sub-group / 개별)

| region key                     | 포함 ring (dof_abbr) | 개수 |
|---------------------------------|----------------------|------|
| `Arm_L`                         | `LSP, LSR, LSY, LEP, LWY, LWR, LWP` | 7 |
| `Arm_L_shoulder`                | `LSP, LSR, LSY` (sub-group)         | 3 |
| `Arm_L_shoulder_pitch`          | `LSP` (개별)                        | 1 |
| `Arm_L_shoulder_roll`           | `LSR`                               | 1 |
| `Arm_L_shoulder_yaw`            | `LSY`                               | 1 |
| `Arm_L_elbow`                   | `LEP`                               | 1 |
| `Arm_L_wrist`                   | `LWY, LWR, LWP` (sub-group)         | 3 |
| `Arm_L_wrist_yaw`               | `LWY`                               | 1 |
| `Arm_L_wrist_roll`              | `LWR`                               | 1 |
| `Arm_L_wrist_pitch`             | `LWP`                               | 1 |
| `Arm_R` 및 `Arm_R_*`            | (위 L → R 대칭, 동일 구조 7+1+1+1+1+3+1+1+1) | — |

##### Hand (2-tier: 전체 / 손가락별)

| region key             | 포함 ring (dof_abbr)                                  | 개수 |
|------------------------|-------------------------------------------------------|------|
| `Hand_L`               | `L11..L14, L21..L24, L31..L34, L41..L44, L51..L54`     | 20   |
| `Hand_R`               | `R11..R14, R21..R24, R31..R34, R41..R44, R51..R54`     | 20   |
| `Hand_L_thumb`         | `L11, L12, L13, L14`                                  | 4    |
| `Hand_L_index`         | `L21, L22, L23, L24`                                  | 4    |
| `Hand_L_middle`        | `L31, L32, L33, L34`                                  | 4    |
| `Hand_L_ring`          | `L41, L42, L43, L44`                                  | 4    |
| `Hand_L_little`        | `L51, L52, L53, L54`                                  | 4    |
| `Hand_R_thumb` ~ `_little` | (R 대칭, 5종)                                       | 4    |

##### 가능한 모든 키 형태 (각 source 별)

```
# 전체
"real"                "sim"

# Arm
"real.Arm_L"          "sim.Arm_L"
"real.Arm_L_shoulder"        "sim.Arm_L_shoulder"
"real.Arm_L_shoulder_pitch"  "sim.Arm_L_shoulder_pitch"
"real.Arm_L_shoulder_roll"   "sim.Arm_L_shoulder_roll"
"real.Arm_L_shoulder_yaw"    "sim.Arm_L_shoulder_yaw"
"real.Arm_L_elbow"           "sim.Arm_L_elbow"
"real.Arm_L_wrist"           "sim.Arm_L_wrist"
"real.Arm_L_wrist_yaw"       "sim.Arm_L_wrist_yaw"
"real.Arm_L_wrist_roll"      "sim.Arm_L_wrist_roll"
"real.Arm_L_wrist_pitch"     "sim.Arm_L_wrist_pitch"
# (Arm_R 도 동일 10종)

# Hand
"real.Hand_L"         "sim.Hand_L"
"real.Hand_L_thumb"   "sim.Hand_L_thumb"
"real.Hand_L_index"   "sim.Hand_L_index"
"real.Hand_L_middle"  "sim.Hand_L_middle"
"real.Hand_L_ring"    "sim.Hand_L_ring"
"real.Hand_L_little"  "sim.Hand_L_little"
# (Hand_R 도 동일 6종)
```

총 키 형태: source 단독 1 + Arm (1 + 1 sub-shoulder + 3 + 1 elbow + 1 sub-wrist + 3) ×
2 sides + Hand (1 + 5) × 2 sides = (1 + 10×2 + 6×2) = **33 per source**, **66 전체**.

**dof_abbr 파싱**:
- `[L|R](1-5)(1-4)` → hand. 1st digit: 1=thumb 2=index 3=middle 4=ring 5=little.
  2nd digit: 1=Yaw/ABAD 2=CMC/MCP 3=MCP/PIP 4=IP/DIP.
- `[L|R](SP|SR|SY|EP|WY|WR|WP)` → arm. SP=Shoulder Pitch, SR=Roll, SY=Yaw,
  EP=Elbow, WY=Wrist Yaw, WR=Roll, WP=Pitch.

#### Region 매칭 (OR 시멘틱)

한 dof 가 여러 region 에 속할 수 있음. 예: LSP 는 `Arm_L`, `Arm_L_shoulder`,
`Arm_L_shoulder_pitch` 셋 다 매칭. 그 중 하나라도 active 면 visible (OR).

```json
"torque_ring_triggers": {
  "real.Arm_L_shoulder": [[10, 20]],
  "real.Arm_L_shoulder_pitch": [[15, 18]]
}
```
- LSP: 10–20s OR 15–18s = 10–20s 동안 visible (Arm_L_shoulder 가 더 큰 범위 cover)
- LSR, LSY: 10–20s 만 (shoulder_pitch 키엔 매칭 안 됨)
- 나머지 real arm dof (LEP, LWY, LWR, LWP): 항상 invisible (exclusive)

(Waist/Neck 은 torque ring 자체가 없어서 region 정의 안 됨.)

### Visibility 규칙 (dof 별 독립, **exclusive 시멘틱**)

```
1) UI 의 torque mode (off/real/sim/both) 가 source 허용 안 함  → invisible
2) source 에 trigger 하나도 없음                              → visible (mode-only default)
3) source 에 trigger 있음:
   3a) dof 매칭됨, True 하나라도                              → visible
   3b) dof 매칭됨, 전부 False                                 → invisible
   3c) dof 매칭 안 됨                                         → invisible  ← exclusive 자동 hide
```

**규칙 1 (UI 모드)**: 사용자가 Visualizer 패널에서 torque mode 를 `off` 로 두면
어떤 trigger 가 박혀 있어도 모든 ring invisible. UI 가 항상 macro switch.

**규칙 2 (비어있는 source)**: `torque_ring_triggers` 섹션 자체가 없거나 `{}` 면
visualizer 의 gate 가 비어있어 mode 만 보고 default visible (UI 만으로 결정).

**규칙 3 (exclusive)**: 한 source 에 region trigger 하나라도 박히면, 그 source 의
**다른 region** dof 는 모두 자동 hide. 예: `real.Hand_L` 만 적으면 real 의 arm·
Hand_R ring 은 자동 invisible. 단 sim 채널엔 영향 없음 (per source 독립).

매칭 gate = `_all_` (source 단독 키, 예: `"real": [...]`) + 그 dof 가 속한 region 들.

### 동작

- replayer 가 매 tick `t_rel` 계산 → 각 (source, region) entry 별 active 평가 →
  transition 시에만 `set_torque_region_gate(src, region, active)` 호출
- replay stop 시 `clear_torque_region_gates()` 로 모두 해제 → user 토글이 다시 단독 결정

### 예시

#### (a) source 단독 — 기존 동작
```json
"torque_ring_triggers": {
  "real": [[33.0, 38.5]]
}
```
모든 real ring 이 33–38.5s 만 visible (mode 가 허용할 때).

#### (b) Region 별 — 일부 영역만 (exclusive)
```json
"torque_ring_triggers": {
  "real.Hand_L":     [[33.0, 38.5]],
  "real.Arm_R":      [[15.0, 25.0]]
}
```
- L 손 ring: 33–38.5s 에만 visible
- R 팔 ring: 15–25s 에만 visible
- **나머지 (R 손, L 팔) 의 real ring: 항상 invisible** (real 에 trigger 가 있는 한
  매칭 안 되는 dof 는 자동 hide). sim 채널은 trigger 없음 → mode 따라 default 동작.

#### (c) 손가락 단위 (exclusive)
```json
"torque_ring_triggers": {
  "real.Hand_L_thumb": [[10, 12]],
  "real.Hand_L_index": [[12, 14]],
  "real.Hand_L_middle":[[14, 16]]
}
```
- L 엄지/검지/중지 ring 만 각자 시간에 visible
- L 약지/새끼, R 손, 팔 등 real 의 다른 모든 ring: 항상 invisible

#### (d) 합집합 (OR 시멘틱, dof 별)
```json
"torque_ring_triggers": {
  "real.Hand_L":       [[20, 30]],
  "real.Hand_L_thumb": [[33, 35]]
}
```
- L thumb ring: 20–30s OR 33–35s 둘 다 visible (두 region 매칭 → OR)
- L index/middle/ring/little ring: 20–30s 만 visible (Hand_L 만 매칭)
- 그 외 real ring (R 손, 팔): 항상 invisible (exclusive)

#### (e-arm) Arm 의 개별 joint 만
```json
"torque_ring_triggers": {
  "real.Arm_L_shoulder_pitch": [[10, 15]],
  "real.Arm_L_elbow":          [[20, 30]]
}
```
- LSP: 10–15s 만 visible
- LEP: 20–30s 만 visible
- 나머지 real arm dof (LSR, LSY, LWY, LWR, LWP), 모든 hand dof: 항상 invisible

#### (f) 일부 region 만 보이고 나머지는 항상
```json
"torque_ring_triggers": {
  "real":          [[0, 9999]],     // 모든 real ring 항상 visible
  "real.Hand_L":   [[33, 38]]        // (의도: L 손만 시간 제한?) — 의미 검토 필요
}
```
이 조합은 OR 시멘틱이라 L 손은 항상 visible (`"real"` gate 가 항상 True 이므로 매칭).
"L 손은 시간제한, 나머지는 항상" 을 원하면 `real` 단독 키 안 쓰고 region 별로 다 적어야 함.

### user mode 와의 관계

- user 가 mode="off" → trigger 무관하게 모든 torque ring invisible
- user 가 mode="real" → sim 채널 trigger 들은 효과 없음 (sim ring 자체가 mode 로 hide)
- user 가 mode="both" + 위 (b) → real 은 region trigger 따라, sim 은 항상 visible

### 모든 torque ring 끄는 방법

```json
"torque_ring_triggers": {
  "real": "off",
  "sim":  "off"
}
```

`"off"` (또는 `false`, `[]`) → 그 source 의 모든 ring 영구 hide. 빈 dict `{}` 와
구분됨 (`{}` 는 "trigger 없음 → mode 기본값 visible").

---

## 4) `ext_joint_torque_triggers` — per-joint external torque 를 ring 에 표시

`/result/<group>/joint/ext_torque` 토픽의 per-joint 외부 토크 값 (Nm 스칼라) 을 ring
에 표시. **기존 torque ring 재사용** — trigger window 동안 ring 값을 internal
torque (`torque_<joint>`) 대신 external torque (`ext_joint_torque_<short>`) 로 **교체**.

> 같은 ring 에 둘 중 하나만 표시되며 충돌 시 external torque 가 이김. 충돌 transition
> 시점에 `[replay] ext_torque substitution active/ended` 로그가 emit 됨 (joint+source
> per window 1회씩, debug 용).
>
> **`torque_ring_triggers` 와의 상호작용**: ext_joint_torque substitution 이 활성인 joint
> 의 ring 은 `torque_ring_triggers` 의 `"off"` / region gate 보다 **우선해서 visible**.
> 즉 `torque_ring_triggers: {"real":"off","sim":"off"}` 로 전체 ring 을 꺼도, 이 섹션에
> 등록된 joint 의 ring 은 ext_torque 값과 함께 표시됨. (UI torque mode 도 동일하게 override.)

### Key 형식

**CSV column name 그대로** — `ext_joint_torque_<short_key>` 한 entry 당 한 joint.
ext_joint_torque CSV 는 real 만 존재하므로 (sim 에는 없음) source 구분 불필요 —
trigger 가 켜진 joint 는 real reader 의 ring 에 substitute 된다.

```json
"ext_joint_torque_triggers": {
  "ext_joint_torque_hand_l_index_abad":     [[0.0, 4.0]],
  "ext_joint_torque_hand_l_index_mcp":      [[0.0, 4.0]],
  "ext_joint_torque_arm_l_shoulder_pitch":  [[5.0, 8.0]],
  "ext_joint_torque_arm_l_elbow":           "off"
}
```

지원되는 short_key 는 `EXT_TORQUE_JOINT_CSV_KEYS` 에 등록된 것 (총 58 joint, follower
포함). CSV header 의 `ext_joint_torque_*` 컬럼명을 그대로 복사하면 됨. 등록되지 않은
short_key 는 WARNING + skip.

> region-level 일괄 지정 (예: "Hand_L 전체") 은 지원 안 함 — 각 joint 명시. 손 전체
> 16개 finger joint 를 한 번에 켜고 싶다면 16 entry 를 같은 range 로 적어야 한다.
> 단순/명시성을 우선한 결정.

> Waist/Neck 은 torque ring 자체가 없어 substitution 효과 없음 (컬럼은 있어도 ring 이
> 없음). 무시됨, warning 없음.

### 우선순위

- trigger window IN + `ext_joint_torque_<short>` 컬럼 존재 → ext_torque × `EXT_JOINT_TORQUE_GAIN_MULT` push
- trigger window OUT → 기존 internal torque push (default 동작)
- key 등록됐지만 컬럼이 CSV 에 없음 → 항상 internal torque (warning 없이 silent fallback)
- key 자체 없음 → 그 joint 는 항상 internal torque

### viz_config.json 연동

`viz_config.json::ext_joint_torque` 섹션:
- `gain_multiplier` (default 1.0): 기존 `torque.gain_*` 위에 곱해지는 보조 스케일.
  ring scale = `clip(|ext_torque × gain_multiplier × per_joint_torque_gain|, MIN, MAX)`.
  per_joint_torque_gain 은 internal torque 기준으로 튜닝됐기 때문에, ext_torque 가
  internal 과 magnitude 단위가 다를 때 (예: contact-induced follower torque 가 작거나
  클 때) ring 크기 보정용.

> 색상 옵션 없음. 별도 prim 안 만들고 기존 torque ring 을 substitute 만 하므로 색은
> 그 ring 의 source(real) 색 (`colors.real`) 그대로. ext_joint_torque CSV 는 real CSV
> 에만 존재 (sim 미발생) — sim 색 역시 의미 없음.

### 로그

```
[replay] ext_torque substitution active @ t=3.142s: joint=L_Index_PIP_Joint, source=real
  — ring 에 internal torque 대신 external torque 가 표시됨
[replay] ext_torque substitution ended @ t=3.200s: joint=L_Index_PIP_Joint, source=real
  — ring 이 internal torque 로 복귀
```

window 진입/이탈 transition 만 emit. 같은 window 안에선 spam 없음.

### CSV 컬럼 확인

```
$ head -1 trajectory/<group>/real_*.csv | tr ',' '\n' | grep '^ext_joint_torque_'
ext_joint_torque_arm_l_shoulder_pitch
ext_joint_torque_arm_l_elbow
ext_joint_torque_hand_l_index_abad
ext_joint_torque_hand_l_index_mcp
ext_joint_torque_hand_l_index_pip
ext_joint_torque_hand_l_index_dip       # ← follower 도 포함됨
ext_joint_torque_hand_l_thumb_yaw
ext_joint_torque_hand_l_thumb_cmc
ext_joint_torque_hand_l_thumb_mcp
ext_joint_torque_hand_l_thumb_ip        # ← thumb follower
ext_joint_torque_neck_yaw               # ← Yaw 가 먼저 (ALLEX_CSV_JOINT_NAMES 는 Pitch-Yaw)
ext_joint_torque_neck_pitch
ext_joint_torque_waist_yaw
ext_joint_torque_waist_lower_pitch
...
```

컬럼 명명 규칙: `ext_joint_torque_<group_short>_<joint_stem_lower>`
- group_short: `arm_l`, `arm_r`, `hand_l_thumb`, `hand_l_index`, ..., `neck`, `waist`
- joint_stem: side/group prefix 와 `_Joint` suffix 떼고 lowercase (예: `Shoulder_Pitch`
  → `shoulder_pitch`, `ABAD` → `abad`)

> 일반 `pos_<short>` / `torque_<short>` 컬럼도 동일한 lowercase/`_Joint`-stripped 규칙
> 사용 (예: `pos_r_ring_abad`, `torque_l_elbow`). 단 group prefix 는 안 붙음.
> 변환 함수: `joint_to_csv_short(joint_full)` in `joint_name_map.py`.

joint 순서는 `src/allex/trajectory_generate/joint_name_map.py::EXT_TORQUE_JOINT_NAMES`,
joint_full → short_csv_key 매핑은 동 파일의 `EXT_TORQUE_JOINT_CSV_KEYS` 참조.
finger/thumb 는 follower (DIP/IP) 포함 4개, neck 은 Yaw-Pitch 순서로
ALLEX_CSV_JOINT_NAMES 와 다름.

> **Wrench torque (`ext_torque_<topic_id>_x`)** 와 prefix 가 명확히 다르다
> (`ext_joint_torque_` vs `ext_torque_`) — 컬럼 grep / showcase_reader 모두에서 충돌 없음.

---

## 시간 결정 흐름

1. **sim 있으면**: 같은 group sim replay → "닿았다" 시점 시각으로 확인
2. **sim 없으면**: `real_*.csv` 의 `ext_force_<id>_*` 컬럼 |F| plot → peak 구간 읽기
3. 살짝 여유 (0.1~0.2 s) 두고 t_on / t_off 정해서 JSON 작성
4. `Real/Sim Replay` 패널 **Refresh → Run** — 매 Run 마다 JSON 재로드. Isaac Sim
   재기동 불필요

## 디버깅

- Run 직후 status 라벨에 `scenario={...}` 표시 → 정상 로드
- 콘솔 INFO 로그:
  ```
  [replay] viz_scenario loaded: force=Nch, ext_torque=Nch, torque_ring=Nch,
                                ext_joint_torque=Nch
  ```
- 잘못된 entry (prefix 안 맞음, range 가 list 아님, t_on >= t_off, 비-숫자, invalid 채널 키 등)
  는 WARNING 한 줄씩 + skip
- ext_joint_torque substitution 진입/이탈 시 WARNING/INFO 로그 (위 §4 참조)
- JSON 자체 invalid → `viz_scenario_config.json parse fail: ...`

## Topic 추가 흐름 요약

### Wrench EE force / torque (force_triggers, ext_torque_triggers)

1. `tools/rosbag_to_csv.py::EXT_FORCE_TOPICS` 에 토픽 path → topic_id 추가
   (force / torque 토픽이 분리돼 있다면 dict 만 분리; 같은 Wrench 면 한 entry 가 양쪽 자동 cover)
2. (선택) `EXT_CONTACT_POS_TOPICS` 에 짝되는 position topic 추가
3. rosbag 재변환 → CSV 에 `ext_force_<id>_*` / `ext_torque_<id>_*` / `contact_pos_<id>_*` 컬럼 생성
4. `viz_scenario_config.json` 에 `force_triggers` / `ext_torque_triggers` key 추가
   (dotted form: `real.<id>` / `sim.<channel>`)
5. Real/Sim Replay 패널 Refresh → Run

sim CSV 의 pair/aggregate force 도 같은 `force_triggers` 안에서 `sim.<channel>` 키로 게이팅.
하드코딩 routing dict 없음 — sim/real 양쪽 동시 표시는 두 entry 작성.

### Per-joint external torque (ext_joint_torque_triggers)

1. controller 가 `/result/<group>/joint/ext_torque` (Float64MultiArray, joint-축 토크) publish
2. `tools/rosbag_to_csv.py` 가 `EXT_TORQUE_JOINT_NAMES` 매핑으로 풀어 CSV 에
   `ext_torque_<joint_full_name>` 스칼라 컬럼 생성 (자동)
3. `viz_scenario_config.json::ext_joint_torque_triggers` 에 region 별 trigger 추가
4. Real/Sim Replay 패널 Refresh → Run

> joint 순서가 바뀌었거나 group 이 추가됐다면 `src/allex/trajectory_generate/joint_name_map.py::EXT_TORQUE_JOINT_NAMES`
> 업데이트 필요. ALLEX_CSV_JOINT_NAMES 와 다른 점: finger/thumb 가 follower(DIP/IP) 포함
> 4개씩, neck 이 Yaw-Pitch 순서.

## 코드 진입점

- 통합 로더: `src/allex/utils/showcase_replay.py::_load_viz_scenario`
- Sim force vector 게이팅 (pair + aggregate): `csv_replayer.py::_push_force_frame_sim`
- Real force arrow 게이팅: `csv_replayer.py::_push_real_force_routed`
- Wrench-torque ring 게이팅: `csv_replayer.py::_push_real_torque_routed` /
  `_set_or_add_torque_ring` / `_set_torque_ring_prim_visible`
- Torque ring trigger 평가: `csv_replayer.py::_update_torque_ring_gates`
- ext_joint_torque substitution: `csv_replayer.py::_maybe_substitute_ext_torque` /
  `_flush_ext_jtq_diff` (transition 로그)
- Trigger 파서:
  - `csv_replayer.py::_parse_force_triggers_v2` — force_triggers / ext_torque_triggers
    (real.* / sim.* dotted form + legacy `ext_force_*` / `ext_torque_*` backward-compat)
  - `csv_replayer.py::_parse_channel_triggers` — torque_ring_triggers
  - `csv_replayer.py::_parse_ext_joint_torque_triggers` — ext_joint_torque_triggers
    (CSV column 그대로 → joint_full reverse-map)
- Visualizer gate API: `force_torque_visualizer.py::set_torque_region_gate` /
  `clear_torque_region_gates` / `_dof_abbr_to_regions` (region 매핑)
- Custom force arrow API:
  `force_torque_visualizer.py::add_custom_force_vector` / `set_custom_force_vector`
- Custom torque ring API (Wrench torque 전용 — free-floating):
  `force_torque_visualizer.py::add_custom_torque_ring` / `set_custom_torque_ring` /
  `_apply_torque_ring_scale` / `_axis_to_quatd`
- Trigger 검증 헬퍼: `csv_replayer.py::_validate_ranges` / `_t_in_ranges`
- viz_config 상수 (Wrench-torque): `EXT_TORQUE_GAIN/MIN_SCALE/MAX_SCALE/HIDE_BELOW`,
  `EXT_TORQUE_COLOR_REAL/SIM`
- viz_config 상수 (per-joint ext_torque): `EXT_JOINT_TORQUE_GAIN_MULT`
  (별도 color 상수 없음 — 기존 ring 의 source color 그대로 재사용)
