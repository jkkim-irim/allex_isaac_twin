# Isaac Sim Newton Force-Access Patches

이 폴더는 ALLEX Twin extension 이 Newton(MuJoCo backend) 의 PD actuator
출력 (`qfrc_actuator`) 을 읽을 수 있도록 **Isaac Sim 소스에 가한
수정본** 을 보관.

자세한 개념/사용법은 [`../docs/newton_force_access.md`](../docs/newton_force_access.md) 참고.

> ⚠️ **현재 제약** (2026-04-23): Newton `request_state_attributes` 화이트리스트
> 가 `mujoco:qfrc_actuator` 만 허용하여 `qfrc_applied` / `qfrc_gravcomp` 는
> **state 로 request 불가** (강제로 시도하면 simulation step 에러 + NewtonStage
> 초기화 실패). 따라서:
> - `control.joint_f` 로 **쓰기는 가능** — solver 내부에 반영되어 physics 에 작용
> - USD `mjc:gravcomp` authoring 으로 **쓰기 가능**
> - **읽기만 불가**: `get_applied_force_feedforward()` / `get_gravcomp_forces()`
>   는 항상 `None` 리턴. 추후 Newton 이 허용 목록 확장하면 자동 동작.

## 대상 버전 (중요)

이 패치는 아래 Isaac Sim 버전을 기준으로 만들어졌음.

| 항목 | 버전 |
|------|------|
| Isaac Sim | **6.0.0-rc.22** (Early Developer Release 2) |
| Git tag | `v6.0.0-dev2` |
| Commit | `befdb80c7bb90e9bcd5460e16436e95498a930fb` (2026-03-16) |
| `isaacsim.physics.newton` ext | `0.6.0` |
| `isaacsim.core.prims` ext | `0.8.8` |
| Newton prebundle | `1.0.0` |

**버전 확인 방법**:
```bash
cat $ISAACSIM_ROOT/VERSION                                # 6.0.0-rc.22
cd $ISAACSIM_ROOT && git describe --tags                  # v6.0.0-dev2
grep "^version" $ISAACSIM_ROOT/source/extensions/isaacsim.physics.newton/config/extension.toml
grep "^version" $ISAACSIM_ROOT/_build/linux-x86_64/release/exts/isaacsim.core.prims/config/extension.toml
```

다른 버전 사용 시: `install.sh --dry-run` 으로 먼저 교체 대상 확인 후,
각 파일을 `diff` 떠서 수정 의도를 유지하면서 수동 머지 권장.

---

## 수정된 파일 (4 개)

```
patches/
├── isaacsim.physics.newton/
│   └── python/impl/
│       ├── newton_stage.py             ← request_state_attributes 3 개 opt-in
│       └── tensors/
│           └── articulation_view.py    ← low-level getter 2 개 추가 + actuator prefer
│
└── isaacsim.core.prims/
    └── isaacsim/core/prims/impl/
        ├── articulation.py             ← multi-env wrapper 2 개 추가
        └── single_articulation.py      ← single-env wrapper 2 개 추가
```

원본 경로 매핑:

| patches 경로 | Isaac Sim 내 원본 경로 |
|-------------|----------------------|
| `isaacsim.physics.newton/python/impl/newton_stage.py` | `source/extensions/isaacsim.physics.newton/python/impl/newton_stage.py` |
| `isaacsim.physics.newton/python/impl/tensors/articulation_view.py` | `source/extensions/isaacsim.physics.newton/python/impl/tensors/articulation_view.py` |
| `isaacsim.core.prims/isaacsim/core/prims/impl/articulation.py` | `_build/linux-x86_64/release/exts/isaacsim.core.prims/isaacsim/core/prims/impl/articulation.py` |
| `isaacsim.core.prims/isaacsim/core/prims/impl/single_articulation.py` | `_build/linux-x86_64/release/exts/isaacsim.core.prims/isaacsim/core/prims/impl/single_articulation.py` |

---

## 설치

### 자동 (install.sh)
```bash
cd patches/
ISAACSIM_ROOT=/home/asher12/workspace/isaacsim_git/isaacsim ./install.sh --backup

# 확인만 해보고 싶으면:
ISAACSIM_ROOT=... ./install.sh --dry-run
```
옵션:
- `--backup` : 기존 파일을 `<name>.orig` 로 백업 후 교체
- `--dry-run` : 복사 없이 어떤 파일이 교체될지만 출력

### 수동
네 파일을 원본 경로로 복사하면 됨. 예:
```bash
cp patches/isaacsim.physics.newton/python/impl/newton_stage.py \
   $ISAACSIM_ROOT/source/extensions/isaacsim.physics.newton/python/impl/newton_stage.py
# (나머지 3개도 동일하게)
```

Isaac Sim 이 이미 켜져 있었다면 **완전 재시작** 필요.

---

## 검증
```bash
grep "qfrc_applied\|qfrc_gravcomp" \
  $ISAACSIM_ROOT/source/extensions/isaacsim.physics.newton/python/impl/newton_stage.py
grep "get_dof_applied_forces\|get_dof_gravcomp_forces" \
  $ISAACSIM_ROOT/source/extensions/isaacsim.physics.newton/python/impl/tensors/articulation_view.py
grep "get_applied_force_feedforward\|get_gravcomp_forces" \
  $ISAACSIM_ROOT/_build/linux-x86_64/release/exts/isaacsim.core.prims/isaacsim/core/prims/impl/articulation.py
grep "get_applied_force_feedforward\|get_gravcomp_forces" \
  $ISAACSIM_ROOT/_build/linux-x86_64/release/exts/isaacsim.core.prims/isaacsim/core/prims/impl/single_articulation.py
```
네 명령 모두 매치가 나와야 정상.

Isaac Sim Script Editor 에서:
```python
from isaacsim.physics.newton import acquire_stage
st = acquire_stage()
attrs = [a for a in dir(st.state_0.mujoco) if "qfrc" in a]
print(attrs)
# ['qfrc_actuator', 'qfrc_applied', 'qfrc_gravcomp'] 세 개 모두 보여야 정상
```

---

## 원복 (백업 사용)

`--backup` 옵션으로 설치했다면:
```bash
for f in $(find $ISAACSIM_ROOT -name "*.orig" 2>/dev/null); do
  mv "$f" "${f%.orig}"
done
```

---

## 주의사항
- Isaac Sim **재빌드 / 업데이트** 시 원본이 복구되어 패치가 날아감. 재적용 필요.
- 이 폴더의 파일들은 **Isaac Sim 공식 소스를 약간만 수정한 버전**이라
  라이선스는 원본(Isaac Sim / NVIDIA SPDX) 그대로 따름.
- 중·장기적으로는 extension 내부에서 **monkey-patch** 하는 방식으로
  교체하면 업데이트에 robust 해짐. 현재는 소스 직접 수정 상태.
