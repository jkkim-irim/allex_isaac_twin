# Newton Force Access — MuJoCo force 분해 항 읽기/쓰기

ALLEX Twin 이 Newton(MuJoCo backend) 에서 joint force 의 각 구성 항을
어떻게 주입/관측하는지 정리한 문서. Isaac Sim Newton extension 소스에
가한 **로컬 패치**도 여기 명시함.

---

## ⚠️ 현재 제약 (중요)

**2026-04-23 확인**: Newton 의 `request_state_attributes()` 는 허용 화이트리스트만
받으며 현재 아래 셋만 허용:

```
body_parent_f, body_qdd, mujoco:qfrc_actuator
```

`mujoco:qfrc_applied` / `mujoco:qfrc_gravcomp` 는 현재 request 불가. 억지로
시도하면 `Unknown extended state attribute` 에러 + NewtonStage 초기화
실패.

### 쓰기/읽기 매트릭스

| 경로 | 쓰기 | 쓴 값 직접 읽기 | solver 적용 후 실제 torque 읽기 |
|------|------|----------------|-------------------------------|
| PD (`ArticulationAction`) | ✅ `apply_action(...)` | ✅ target 기억 | ✅ `get_applied_joint_efforts()` |
| FF (`control.joint_f`) | ✅ `.assign(arr)` | ✅ `.numpy()` 로 덤프 (pass-through) | ❌ 현재 불가 |
| Gravcomp (`mjc:gravcomp`) | ✅ USD attr | ✅ per-body 계수 조회 | ❌ 현재 불가 |

### FF 시각화 방법 — extension 의 FeedforwardTorqueManager

`control.joint_f` 는 입력 = 적용값 pass-through 구조라 쓴 값 보관만
하면 "sim 이 받은 actuation" 으로 쓸 수 있음. 이를 위해 extension 내부에
**`FeedforwardTorqueManager`** (`src/core/feedforward_torque.py`) 제공.

```python
# 주입 쪽 (control loop)
ff_mgr = scenario.get_ff_manager()
ff_mgr.apply(tau_array)      # control.joint_f 로 쓰기 + 내부 버퍼 저장

# 관측 쪽 (visualizer / logger)
ff_mgr = scenario.get_ff_manager()
tau_cur = ff_mgr.get_last()  # 최근 apply 로 쓴 float32 배열 (copy)
```

특징:
- 첫 `apply()` 호출 때 `num_dof` 자동 감지. 또는 `set_num_dof(60)` 으로
  사전 지정 가능. scenario 가 articulation attach 시점에 자동으로 맞춰 줌.
- 실패해도 (Newton 미준비 등) **내부 버퍼 는 항상 갱신** 됨 → 쓴 값 tracking 은 보장.
- `clear()` 로 0 초기화 (trajectory 종료 시 residual torque 제거).

> 직접 `st.control.joint_f.numpy()` 로 덤프하는 방법도 가능하지만, 매니저를
> 쓰면 scenario 에서 단일 포인트로 관리되고 (시작/종료 시 clear 등) 추후
> Newton 이 qfrc_applied 를 expose 하도록 바뀌어도 매니저 내부만 교체하면 됨.

### Gravcomp 는 sim 쪽에서 실제 torque 불가

입력(`mjc:gravcomp`)은 per-body 스칼라 계수, 출력은 per-joint Nm. solver
가 dynamics 역산해서 변환하는데 그 결과를 현재 읽을 수 없음.

대안:
- **권장**: gravcomp 안 쓰고 real robot 의 `joint_trq_target` 을 통째로
  FF 로 주입 (§5.1). actuation 전체가 FF 한 슬롯이라 tracking 100%.
- 체념: "gravcomp ON" 플래그만 표시.
- 재계산: Newton-Euler 라이브러리로 별도 계산 (구현 복잡).

### 현재 로컬 패치 범위

`qfrc_applied` / `qfrc_gravcomp` 관련 wrapper 는 dead code 회피 위해
**원복한 상태**. 현재 패치는 **`qfrc_actuator` 만** 건드림:

| 파일 | 변경 |
|------|------|
| `newton_stage.py` | `request_state_attributes("mujoco:qfrc_actuator")` 1 줄 추가 |
| `articulation_view.py` | `get_dof_actuation_forces()` 내부 로직: `qfrc_actuator` prefer, fallback `joint_torques` |
| `articulation.py`, `single_articulation.py` | **변경 없음** (기존 `get_applied_joint_efforts()` 만 사용) |

### 향후 Newton 이 화이트리스트 확장하면

1. `newton_stage.py` 에 두 줄 추가:
   ```python
   self.model.request_state_attributes("mujoco:qfrc_applied")
   self.model.request_state_attributes("mujoco:qfrc_gravcomp")
   ```
2. 필요하면 `articulation_view.py` 에 `get_dof_applied_forces()` /
   `get_dof_gravcomp_forces()` + `articulation.py` / `single_articulation.py`
   wrapper 들을 재추가.
3. Git 히스토리에 이전 버전 구현 남아있음 (이 문서가 참고).

---

## 0. 대상 버전

이 문서와 `patches/` 는 아래 Isaac Sim 버전을 기준으로 작성됨.
API 시그니처나 내부 구조는 버전마다 달라질 수 있으니 반드시 동일
또는 매우 근접한 버전에서만 적용할 것.

| 항목 | 버전 |
|------|------|
| Isaac Sim | **6.0.0-rc.22** (Early Developer Release 2) |
| Git tag | `v6.0.0-dev2` |
| Commit | `befdb80c7bb90e9bcd5460e16436e95498a930fb` (2026-03-16) |
| `isaacsim.physics.newton` ext | `0.6.0` |
| `isaacsim.core.prims` ext | `0.8.8` |
| Newton prebundle | `1.0.0` |
| MuJoCo backend | MuJoCo Warp (번들됨) |

> 다른 버전에서 쓰려면:
> 1. `patches/` 파일들이 원본과 머지 가능한지 먼저 `diff` 로 확인
> 2. 충돌 나는 부분은 §4 의 수정 의도를 참고해 수동 반영
> 3. Newton API 자체(`request_state_attributes`, `state_0.mujoco.qfrc_*`,
>    `control.joint_f`)가 바뀌면 구조적으로 재설계 필요

---

## 1. MuJoCo force 분해 (개념)

Newton 은 MuJoCo Warp solver 를 사용한다. 매 physics step 에서 joint 에
적용되는 총 generalized force 는 **서로 독립된 슬롯의 합** 이다.

```
joint_acceleration = M⁻¹ · [ qfrc_actuator
                            + qfrc_applied
                            + qfrc_gravcomp
                            - qfrc_bias          (Coriolis / 중력 / passive)
                            + constraint_force ]
```

| 슬롯 | 의미 | 채워지는 경로 |
|------|------|---------------|
| `qfrc_actuator` | **PD actuator** 계산 결과 `kp·(q_ref-q) + kd·(qd_ref-qd)` | `articulation.apply_action(ArticulationAction(positions, velocities))` |
| `qfrc_applied`  | **외부 피드포워드 torque** (user input) | `control.joint_f.assign(...)` |
| `qfrc_gravcomp` | **자동 중력 상쇄** (body 별 `gravcomp` 계수로 계산) | USD 속성 `mjc:gravcomp` authoring |
| `qfrc_bias`     | Coriolis + 중력 + passive spring/damper | solver 내부 자동 |

> 세 슬롯(`qfrc_actuator`, `qfrc_applied`, `qfrc_gravcomp`)은 **중복 가산** 된다.
> 실 로봇의 total command 를 sim 에 재현하려면 **하나의 슬롯에만 몰아넣거나**
> 사용하는 슬롯을 명확히 분리해야 한다.

---

## 2. 쓰기 (제어 입력)

### 2.1 PD actuator (`qfrc_actuator`)
```python
# target position 을 articulation 에 apply
from isaacsim.core.utils.types import ArticulationAction
articulation.apply_action(ArticulationAction(joint_positions=target_positions))
```
- `_MJCF_DRIVE_GAINS` (asset_manager.py) 가 kp/kd 세팅을 overrride
- `kp=0, kd=0` 으로 세팅하면 PD 비활성 → 완전 torque 모드

### 2.2 Feedforward torque (`control.joint_f` → `qfrc_applied`)

**권장 (extension 내 매니저 사용)**
```python
ff_mgr = scenario.get_ff_manager()
ff_mgr.apply(ff_torque_array)    # 쓰기 + 최근 값 저장
last = ff_mgr.get_last()         # 나중에 visualizer 등에서 조회
```
- `src/core/feedforward_torque.py` 의 `FeedforwardTorqueManager` 가
  `control.joint_f.assign()` 을 감싸 주입 + 내부 버퍼에 snapshot 보관.
- Newton 이 qfrc_applied 를 read-back 못하게 막고 있으므로 (§제약 섹션)
  매니저가 **사실상 ground truth** 역할.

**Low-level (직접 접근 시)**
```python
from isaacsim.physics.newton import acquire_stage
st = acquire_stage()
if st.control.joint_f is not None:
    st.control.joint_f.assign(ff_torque_array)  # length = joint_dof_count
```
- Newton 의 `Control` 클래스 **표준 필드**. 라이브러리 수정 불필요.
- 매 physics step 마다 필요하면 갱신 (step 후 내부 reset 될 수 있음).
- MuJoCo solver 가 이 값을 `mj_data.qfrc_applied` 로 복사 (solver_mujoco.py:3239).

### 2.3 Gravity compensation (`mjc:gravcomp` → `qfrc_gravcomp`)
```python
from pxr import Sdf
attr = body_prim.CreateAttribute("mjc:gravcomp", Sdf.ValueTypeNames.Float, True)
attr.Set(1.0)   # 1.0 = 완전 상쇄, 0.0 = 없음, 부분 값 가능
```
- 또는 MJCF 에 `<body gravcomp="1.0">` authoring 후 USD convert.
- **Body 별** 설정 (per-body coefficient). Newton 이 dynamics 역산해서
  해당 body 가 느끼는 중력을 상쇄하는 joint torque 를 매 step 자동 계산.
- 라이브러리 수정 불필요.

---

## 3. 읽기 (관측)

### 3.1 요약 테이블 (SingleArticulation 기준)
| 내용 | Python API | 현재 |
|------|-----------|------|
| PD actuator 출력 (qfrc_actuator) | `articulation.get_applied_joint_efforts()` | ✅ `(num_dof,)` tensor |
| FF (qfrc_applied) | — 전용 getter 없음 | ⚠️ `st.control.joint_f.numpy()` 로 자기가 쓴 값 되읽기 (pass-through) |
| Gravcomp (qfrc_gravcomp) | — 전용 getter 없음 | ⚠️ 현재 sim 에서 실제 torque 못 읽음 |

> Newton 이 화이트리스트 확장 시 extension 쪽에 FF/gravcomp getter 추가
> 계획. 현재는 PD 한 경로만 지원.

### 3.2 데이터 흐름 (현재)
```
MuJoCo Warp solver (physics step 내부)
  ├─ qfrc_actuator  (PD 결과)        ← 현재 노출 O
  ├─ qfrc_applied   (control.joint_f 복사본)  ← 노출 X (화이트리스트 제약)
  └─ qfrc_gravcomp  (gravcomp 결과)          ← 노출 X
       │
       ▼ (qfrc_actuator 만 opt-in request)
Newton State (warp array on GPU)
  state_0.mujoco.qfrc_actuator
       │
       ▼
isaacsim.physics.newton — NewtonArticulationView (= _physics_view)
  get_dof_actuation_forces()    → qfrc_actuator (prefer 로직 패치됨)
       │
       ▼
isaacsim.core.prims — Articulation / SingleArticulation
  get_applied_joint_efforts()   → qfrc_actuator (기존 Isaac Sim API)
       │
       ▼
allex_isaac_twin 사용 코드
  articulation.get_applied_joint_efforts()       # PD

# FF 추적은 별도 경로:
  st = acquire_stage()
  st.control.joint_f.assign(ff_array)   # 쓰기
  st.control.joint_f.numpy()            # 되읽기 (= actuation, pass-through)
```

### 3.3 Total torque 계산 예시

현재 FF/gravcomp 는 `None` 리턴하므로 PD 만 합산됨. FF 시각화가 필요하면
Python 쪽에서 `control.joint_f` 에 쓴 값을 별도 변수로 보관해 합산.

```python
from isaacsim.physics.newton import acquire_stage
import numpy as np

qa = articulation.get_applied_joint_efforts()     # PD (qfrc_actuator)
qf = articulation.get_applied_force_feedforward() # FF: 현재 None
qg = articulation.get_gravcomp_forces()           # gravcomp: 현재 None

total = np.asarray(qa, dtype=np.float32)

# FF 합산 — Newton 이 expose 안 하니 control 에서 되읽기 (pass-through)
st = acquire_stage()
if st is not None and st.control is not None and st.control.joint_f is not None:
    total = total + np.asarray(st.control.joint_f.numpy(), dtype=np.float32)

# gravcomp 는 현재 재구성 불가 — Newton 화이트리스트 확장 시 qg 사용
if qg is not None:
    total = total + np.asarray(qg, dtype=np.float32)

# total: joint 에 실제 적용되는 actuation 총합 (bias/constraint/gravcomp 제외)
```

---

## 4. Isaac Sim 로컬 패치 (필수)

`qfrc_actuator` 를 읽기 위해 Isaac Sim 소스 **두 파일**에 최소 수정.
(FF/gravcomp wrapper 는 Newton 화이트리스트 확장 전까지 dead code 라
모두 제거한 상태.)

### 4.1 `isaacsim.physics.newton/python/impl/newton_stage.py`
`NewtonStage.initialize_newton()` 내 `self.model.builder.finalize(...)` 직후:
```python
# ALLEX local patch: PD actuator 결과를 state 에 노출 (opt-in).
# qfrc_applied / qfrc_gravcomp 는 Newton 의 허용 화이트리스트
# (body_parent_f, body_qdd, mujoco:qfrc_actuator) 에 없어 request 불가.
self.model.request_state_attributes("mujoco:qfrc_actuator")
```

이걸 안 하면 `state_0.mujoco.qfrc_actuator` 속성이 `None`.

> ⚠️ `mujoco:qfrc_applied` / `mujoco:qfrc_gravcomp` 는 현재 Newton 에서
> request 불가. 억지로 추가하면 simulation step 에러 후 NewtonStage 가
> 초기화 실패함 (`AttributeError: 'NewtonStage' object has no attribute 'control'`).
> Newton 업데이트 후 허용 목록 확장되면 이 부분에 두 줄 추가로 활성화 가능.

### 4.2 `isaacsim.physics.newton/python/impl/tensors/articulation_view.py`
- `get_dof_actuation_forces()` 의 **내부 로직만 변경**: `qfrc_actuator` 우선
  리턴. 원래는 `joint_torques` 만 리턴했지만 이는 `set_dof_actuation_forces()`
  호출 시에만 채워짐 → position PD 모드에서는 항상 0 이라 시각화 불가.
- API 시그니처는 원본과 동일. `articulation.get_applied_joint_efforts()`
  호출 경로가 그대로 동작하면서 PD 결과를 보여주게 됨.

### 4.3 `isaacsim.core.prims` 파일들
- **변경 없음**. 기존 `get_applied_joint_efforts()` 만 사용 (Isaac Sim
  기본 API).

### 4.4 패치 유지 주의사항
- Isaac Sim 재빌드 / 업데이트 시 **덮어씌워질 수 있음**. 변경된 파일:
  1. `isaacsim.physics.newton/.../newton_stage.py` (opt-in 한 줄)
  2. `isaacsim.physics.newton/.../tensors/articulation_view.py`
     (`get_dof_actuation_forces` 내부 로직)
- 수정된 파일 사본은 **`patches/`** 폴더에 보관. 재적용은 §4.5 참고.
- 중·장기적으론 extension 쪽에서 **monkey-patch** 하는 게 업그레이드에
  안전. 현재는 소스 직접 수정 상태.

### 4.5 패치 적용 (다른 사람 / 재빌드 후)

이 repo 의 `patches/` 폴더에 수정된 버전이 담겨 있음. 자동 설치:
```bash
cd patches/
ISAACSIM_ROOT=/path/to/isaacsim ./install.sh --backup

# 변경 없이 어떤 파일이 교체될지만 확인:
ISAACSIM_ROOT=... ./install.sh --dry-run
```

`--backup` 옵션은 기존 파일을 `<원본>.orig` 로 보존 후 교체. 자세한 사용법
및 수동 복사 방법은 [`patches/README.md`](../patches/README.md) 참고.

**적용 확인**:
```bash
grep "mujoco:qfrc_actuator" \
  $ISAACSIM_ROOT/source/extensions/isaacsim.physics.newton/python/impl/newton_stage.py
grep "qfrc_actuator" \
  $ISAACSIM_ROOT/source/extensions/isaacsim.physics.newton/python/impl/tensors/articulation_view.py
```
두 명령 모두 매치가 나와야 정상. Isaac Sim 완전 재시작 후 사용 가능.

---

## 5. 실 로봇과의 매칭을 위한 권장 구성

### 5.1 가장 clean 한 방식 — real torque 를 FF 로 주입
```
Newton PD       : disable (kp = kd = 0)
mjc:gravcomp    : 0 (off)
control.joint_f : rosbag 의 joint_trq_target 값 그대로
```
- Sim 입력 = 실 로봇이 motor driver 에 보낸 총 command
- 차이 원인이 **dynamics 모델 (mass/inertia/friction)** 으로 좁혀짐
- Visualizer 는 `get_dof_applied_forces()` 를 읽어야 함

### 5.2 Position PD + gravcomp (덜 정확)
```
Newton PD       : rosbag joint_ang_target_deg 를 target 으로
mjc:gravcomp    : 1.0 (자동)
control.joint_f : 안 씀
```
- Gravcomp 수식이 실 로봇과 다를 수 있어 논문에서 설득력 약함
- Visualizer 는 `get_dof_actuation_forces() + get_dof_gravcomp_forces()`

### 5.3 피해야 할 구성
```
Position PD + gravcomp + joint_trq_target FF 동시
```
→ **중복 가산** (PD + grav + FF 모두 더해짐) → sim 이 실 로봇보다 훨씬
   큰 torque 받음 → 발산 또는 이상 거동.

---

## 6. 검증 체크리스트

Script Editor 에서 Play 상태로 실행해 확인할 것들:

```python
from isaacsim.physics.newton import acquire_stage
st = acquire_stage()

# 1) opt-in 확인: 현재 qfrc_actuator 만 노출됨
attrs = [a for a in dir(st.state_0.mujoco) if "qfrc" in a]
print(attrs)
# 기대: ['qfrc_actuator', ...]  (qfrc_applied/gravcomp 는 화이트리스트 제약)
```

**현재 가능한 검증**
- [ ] `dir(state_0.mujoco)` 에 `qfrc_actuator` 존재 (그 외 `qfrc_*` 는 미노출)
- [ ] PD target 을 현재 pose 와 크게 다르게 주면
      `get_applied_joint_efforts()` 에 PD error 크기만큼 nonzero 값
- [ ] `ff_mgr.apply(tau)` 후 `ff_mgr.get_last()` 가 동일한 배열 리턴
- [ ] `ff_mgr.apply(tau)` 이후 로봇이 그 방향으로 움직임 (physics 반영 확인)

**현재 불가 (Newton 화이트리스트 제약)**
- solver 가 qfrc_applied / qfrc_gravcomp 에 넣은 값을 sim 에서 되읽기
- 즉 FF 는 extension 측 `ff_mgr.get_last()` 가 유일한 ground truth

---

## 7. 파일 트리 참조

```
allex_isaac_twin/
├── docs/
│   └── newton_force_access.md        ← 이 문서
├── patches/
│   ├── README.md                     ← 패치 설치 안내
│   ├── install.sh                    ← 자동 복사 스크립트
│   ├── isaacsim.physics.newton/
│   │   └── python/impl/
│   │       ├── newton_stage.py
│   │       └── tensors/articulation_view.py
│   └── isaacsim.core.prims/
│       └── isaacsim/core/prims/impl/
│           ├── articulation.py
│           └── single_articulation.py
└── src/
    └── core/
        ├── force_torque_visualizer.py  ← 현재 qfrc_actuator 만 읽음
        │                                 (FF 시각화 하려면 ff_manager.get_last() 추가)
        └── feedforward_torque.py       ← FeedforwardTorqueManager 
                                          (control.joint_f 주입 + 쓴 값 tracking)
```

`scenario.get_ff_manager()` 로 접근.
