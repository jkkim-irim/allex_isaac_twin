# Real/Sim Replay — Viz Scenario Config 사용법

`Real/Sim Replay` 패널의 force vector / torque ring / graph plot 시각화를 시간 기반
시나리오로 제어하는 통합 annotation 파일.

## 파일 위치

각 group 디렉토리 안에 한 개:

```
trajectory/<group>/
├── real_*.csv                    ← rosbag_to_csv.py 가 생성, 수정 X
├── sim_*.csv                     ← 있어도 되고 없어도 됨
└── viz_scenario_config.json      ← (선택) 손으로 작성
```

**파일이 없으면 기존 default 동작**: force 는 sim-active gate, torque ring 은 user 토글,
graph plot 없음.

## JSON 스키마 (전부 optional)

```json
{
  "_notes": "Real/Sim Replay 시나리오 — 시간 단위 [s] (rosbag 첫 sample 0 기준 상대시간)",

  "force_triggers": {
    "ext_force_arm_r":        [[33.0, 38.5]],
    "ext_force_hand_l_thumb": [[33.0, 35.5]]
  },

  "torque_ring_triggers": {
    "real": [[10.0, 50.0]],
    "sim":  [[10.0, 50.0]]
  },

  "graph_plots": [
    /* Phase 3 — 미구현. 작성해도 현재는 무시 (보관만 됨). */
  ]
}
```

`_` 로 시작하는 키 (예 `_notes`) 는 무시 (메모용).

---

## 1) `force_triggers` — Real ext_force vector 시간 게이팅

### Key 결정

real CSV header 의 `ext_force_<id>_x/y/z` 컬럼 **공통 prefix** 그대로 사용. 일반 패턴
`ext_force_<topic_id>`. `ext_force_` 가 아닌 prefix 키는 무시 + WARNING.

#### 현재 등록된 모든 key (총 14)

`tools/rosbag_to_csv.py::EXT_FORCE_TOPICS` 기준:

| JSON key                  | CSV 컬럼                          | 설명                       |
|---------------------------|-----------------------------------|----------------------------|
| `ext_force_arm_l`         | `ext_force_arm_l_{x,y,z}`         | L_Palm 외력                |
| `ext_force_arm_r`         | `ext_force_arm_r_{x,y,z}`         | R_Palm 외력                |
| `ext_force_shoulder_l`    | `ext_force_shoulder_l_{x,y,z}`    | L Shoulder reaction       |
| `ext_force_shoulder_r`    | `ext_force_shoulder_r_{x,y,z}`    | R Shoulder reaction       |
| `ext_force_hand_l_thumb`  | `ext_force_hand_l_thumb_{x,y,z}`  | L 엄지 외력                |
| `ext_force_hand_l_index`  | `ext_force_hand_l_index_{x,y,z}`  | L 검지 외력                |
| `ext_force_hand_l_middle` | `ext_force_hand_l_middle_{x,y,z}` | L 중지 외력                |
| `ext_force_hand_l_ring`   | `ext_force_hand_l_ring_{x,y,z}`   | L 약지 외력                |
| `ext_force_hand_l_little` | `ext_force_hand_l_little_{x,y,z}` | L 새끼 외력                |
| `ext_force_hand_r_thumb`  | `ext_force_hand_r_thumb_{x,y,z}`  | R 엄지 외력                |
| `ext_force_hand_r_index`  | `ext_force_hand_r_index_{x,y,z}`  | R 검지 외력                |
| `ext_force_hand_r_middle` | `ext_force_hand_r_middle_{x,y,z}` | R 중지 외력                |
| `ext_force_hand_r_ring`   | `ext_force_hand_r_ring_{x,y,z}`   | R 약지 외력                |
| `ext_force_hand_r_little` | `ext_force_hand_r_little_{x,y,z}` | R 새끼 외력                |

#### 신규 topic 추가

`tools/rosbag_to_csv.py::EXT_FORCE_TOPICS` 에 새 topic 추가 후 CSV 재생성하면, JSON 에
`ext_force_<topic_id>` 키만 박으면 자동 visualize. routing dict (`SIM_TO_REAL_FORCE_ROUTE`)
수정 없이도 trigger 만으로 OK.

### 우선순위 (topic 별 독립, torque_ring_triggers 와 동일 시멘틱)

1. **`force_triggers` 에 이 topic 의 key 있음** → 시간 범위로 on/off (sim 없이 동작)
2. **`force_triggers` 가 비어있지 않은데 이 topic 만 미정의** → **exclusive hide**
   (다른 topic 만 명시한 trigger 의 부수효과)
3. **`force_triggers` 통째로 비어있음 + sim CSV 있음 + routing 매핑 있음** →
   sim `|F|>1mN` active gate
4. **`force_triggers` 통째로 비어있음 + sim 없음** → default visible
   (mode-only, threshold-hide 가 작은 force 알아서 숨김)

| 이 topic key | 다른 topic key | sim CSV | routing 매핑 | 결과 게이트         |
|--------------|----------------|---------|-------------|---------------------|
| ✅           | 무관           | 무관    | 무관        | trigger 시간 범위    |
| ❌           | ✅ (다른 키 있음) | 무관    | 무관        | 항상 off (exclusive) |
| ❌           | ❌ (전부 비어있음) | ✅      | ✅          | sim `\|F\|>1mN`      |
| ❌           | ❌ (전부 비어있음) | ✅      | ❌          | skip (gate 신호 없음) |
| ❌           | ❌ (전부 비어있음) | ❌      | 무관        | **default visible**  |

→ **section 통째로 비어있고 sim 도 없으면 모든 ext_force topic 이 default visible**
(threshold-hide 미만은 자동 숨김). 즉 real-only group 에서 trigger 안 적어도 force
가 일단 다 보임.

### Visualization 파이프라인

```
vector  = ext_force_<topic_id>_{x,y,z}     ← real CSV (chest_origin frame)
origin  = contact_pos_<topic_id>_{x,y,z}   ← real CSV (chest_origin frame)
        → chest→world transform (sim Chest_Origin_Link world matrix, Fabric API)
prim_path = /World/AllexForceViz/real/<topic_id>
```

`contact_pos_<topic_id>_*` 컬럼이 없으면 chest_origin link world 위치로 fallback.
UI "Real arrow @ sim contact" 토글 ON + sim 가용 + sim active 채널 매칭 시 origin 만 sim
contact_pos 로 override.

---

## 2) `torque_ring_triggers` — Torque ring 시간/영역 게이팅

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

---

## 3) `graph_plots` — Phase 3 미구현

스키마 자리만 잡아둠. 현재 작성해도 무시됨 (load 시 INFO 로그에 plots=N 만 표시).

설계 메모는 `dvcc/21_viz_scenario_config_design.md` 의 Phase 3 섹션 참조.

---

## 시간 결정 흐름

1. **sim 있으면**: 같은 group sim replay → "닿았다" 시점 시각으로 확인
2. **sim 없으면**: `real_*.csv` 의 `ext_force_<id>_*` 컬럼 |F| plot → peak 구간 읽기
3. 살짝 여유 (0.1~0.2 s) 두고 t_on / t_off 정해서 JSON 작성
4. `Real/Sim Replay` 패널 **Refresh → Run** — 매 Run 마다 JSON 재로드. Isaac Sim
   재기동 불필요

## 디버깅

- Run 직후 status 라벨에 `scenario={force=Nch, torque=Nch, plots=N}` 표시 → 정상 로드
- 콘솔 INFO 로그: `[replay] viz_scenario loaded: force=N..., torque=N..., plots=N`
- 잘못된 entry (prefix 안 맞음, range 가 list 아님, t_on >= t_off, 비-숫자, invalid 채널 키 등)
  는 WARNING 한 줄씩 + skip
- JSON 자체 invalid → `viz_scenario_config.json parse fail: ...`

## Topic 추가 흐름 요약 (force_triggers)

1. `tools/rosbag_to_csv.py::EXT_FORCE_TOPICS` 에 토픽 path → topic_id 추가
2. (선택) `EXT_CONTACT_POS_TOPICS` 에 짝되는 position topic 추가
3. rosbag 재변환 → CSV 에 `ext_force_<id>_*` / `contact_pos_<id>_*` 컬럼 생성
4. `viz_scenario_config.json` 의 `force_triggers` 에 `"ext_force_<id>": [[t_on, t_off]]` 추가
5. Real/Sim Replay 패널 Refresh → Run

`SIM_TO_REAL_FORCE_ROUTE` 수정은 sim-active fallback gate (sim only replay 시 자동 visualize)
필요할 때만.

## 코드 진입점

- 통합 로더: `src/allex/utils/showcase_replay.py::_load_viz_scenario`
- 게이팅 / FK 변환: `csv_replayer.py::_push_real_force_routed`
- Torque ring trigger 평가: `csv_replayer.py::_update_torque_ring_gates`
- Visualizer gate API: `force_torque_visualizer.py::set_torque_region_gate` /
  `clear_torque_region_gates` / `_dof_abbr_to_regions` (region 매핑)
- Trigger 검증 헬퍼: `csv_replayer.py::_validate_ranges` / `_t_in_ranges`
- Routing fallback (force sim-active gate only): `csv_replayer.py::SIM_TO_REAL_FORCE_ROUTE`
