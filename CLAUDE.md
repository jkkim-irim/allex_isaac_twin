# CLAUDE.md

Claude Code가 이 리포에서 작업할 때 참조할 규약. 운영/셋업 절차는 `README.md`에 있으니 여기는 **코드에서 유추 불가능한 invariant와 규약**만 기록.

---

## 한 줄 요약

**Real2Sim 디지털 트윈** — Isaac Sim 6.0 + Newton(MuJoCo Warp) 백엔드로 ALLEX 휴머노이드 구동. ROS2 도메인의 실제 로봇 관절을 200 Hz로 미러링하거나, `trajectory/<group>/*.csv` 그룹을 Hermite 스플라인으로 재생. Traj Studio UI 안에서 PD 게인/토크 리밋을 via 이벤트로 실시간 ramp.

자세한 스펙(LOAD/RUN 절차, CSV 포맷, ROS2 토픽 매핑)은 `README.md` 참조.

---

## 불변식 (변경 시 사용자 확인 필수)

- **`asset/ALLEX/ALLEX.usd` 가 kinematics·inertia·MJC schema(`mjc:gravcomp`, `mjc:actuatorgravcomp`)의 단일 소스**. MJCF는 import 후 USD로 굳어지므로 런타임 동작은 USD만 본다. USD 변경 후 reload 필요.
- **`src/allex/config/physics_config.json` 이 모든 물리 오버라이드의 단일 진입점.** NewtonConfig·MuJoCoSolverConfig·USD `/physicsScene`·gravcomp body 리스트를 여기서 한 번에 정의. 적용 로직은 `src/allex/utils/sim_settings_utils.py` (`apply_newton_config` / `apply_gravcomp_to_builder` / `apply_usd_physics_scene`). 새 파라미터를 다른 곳에서 직접 setattr 하지 말 것.
- **`pd_scale = π/180`** — Newton USD importer가 USD drive 게인에 deg→rad를 한 번 더 곱하는 걸 상쇄. 1.0으로 되돌리면 게인이 ~57.3× 폭주. 주석으로 박혀 있어도 의미 모르고 건드리지 말 것.
- **`use_cuda_graph = True`** 유지 — 매 스텝 외부에서 model array를 새로 할당/주소교체하면 그래프 재기록으로 stutter. **데이터 array(qfrc_applied 등)에 contents만 쓰는 건 안전**. zero-copy torch view를 쓰는 trajectory_player의 PD ramp 패턴을 그대로 따를 것.
- **`mjc:gravcomp` 인증된 body 1개라도 빌드 타임에 있어야** `m.ngravcomp > 0`이 되어 보상 커널이 돈다. 런타임에 array에만 1.0을 써넣어도 ngravcomp는 안 바뀌므로 보상 안 됨. 새 body 추가는 `physics_settings.py::_CONFIG["newton"]["gravcomp"]["bodies"]` 리스트로.
- **Equality constraint는 `newton_bridge.py`가 finalize 후크에서 주입.** Newton USD importer가 MJCF `<equality>`를 누락하므로, `src/allex/config/joint_config.json::equality_constraints`에서 follower/master/polycoef 세트로 보강. JSON은 `tools/regen_joint_config.py`로 MJCF에서 자동 재생성하지만 `ui`·`drive_gains` 섹션은 손작업으로 보존.
- **`active_joints` (joint_config.json) 순서는 trajectory CSV의 joint_1..N 인덱스와 직결**. 재정렬은 모든 trajectory CSV 재생성을 강제하는 파괴적 변경.
- **단위 변환은 경계 레이어에서만**: trajectory CSV position=deg → 내부 rad 변환은 `joint_controller`/`trajectory_player`. PD 게인·토크 리밋은 CSV 원본 단위(SI) 그대로 통과. ROS2 outbound `joint_positions_deg`는 deg, 내부는 rad.
- **Showcase CSV `torque_*` 컬럼 = `qfrc_actuator + qfrc_gravcomp`** (PD + 중력보상 합산). rosbag `joint_torque`(컨트롤러 내부 보상 포함된 모터 명령)와 같은 reference frame. 단독 PD/단독 보상 보고 싶으면 `core/gravcomp_debug.py::GravcompTorqueProbe`.

---

## 환경 규약

- **Conda env `isaacsim` (Python 3.12) 안에서만 작업**. base 환경의 Python으론 USD/Warp 임포트 깨진다.
- **Isaac Sim 실행 전에 절대 `source /opt/ros/*/setup.bash` 하지 말 것.** Isaac Sim 내장 ROS2 브리지와 시스템 ROS2가 충돌 → 코어 덤프. ROS2는 Isaac Sim이 자체적으로 제공한다.
- 본 extension은 `$CONDA_PREFIX/lib/python3.12/site-packages/isaacsim/extsUser/allex_isaac_twin/` 경로에 위치해야 Isaac Sim Extension Manager가 인식. `cdisaac` 별칭으로 빠르게 이동.
- 코드 변경 후 적용은 Extension Manager에서 disable→enable. 캐시(`__pycache__`) 살아있으면 한 번 disable 만으론 안 잡히는 경우 있으니 꼬이면 isaacsim 재기동.
- `mcap` / `mcap_ros2` 는 conda env에 이미 설치돼 있음 — rosbag 파싱 스크립트는 별도 설치 없이 동작.

---

## 아키텍처 원칙

- **`extension.py` → `scenario.py` → `core/*` 의 단방향 호출.** UI(`ui_builder.py`, `ui_utils/*`)는 scenario를 통해서만 core를 건드림. core가 UI를 import 하지 않는다.
- **Newton stage 핸들은 `from isaacsim.physics.newton import acquire_stage` 로 매번 acquire.** 캐시해두지 말 것 — disable/reload 후 stale 참조가 stuck physics를 유발.
- **`builder.custom_attributes["mujoco:*"]` 는 `register_custom_attributes` + `add_usd` 다음 / `_orig_finalize` 이전에만 안전하게 쓸 수 있음.** 그 외 시점은 attribute가 미등록이거나 model이 이미 frozen.
- **Showcase logger의 pose-aligned start 는 align rosbag 기준으로 RMS<0.05 rad.** `_ALIGN_ROSBAG = None` 이면 즉시 기록 모드로 폴백. 다른 rosbag으로 비교하려면 그 상수만 교체.
- **GPU array 직접 쓰기는 zero-copy view 기반으로.** `solver._update_joint_dof_properties()` 같이 명시 동기화 함수가 있을 땐 매 스텝 호출 (MuJoCo gainprm/biasprm 동기). torch tensor → warp array view는 graph capture 친화적.
- **trajectory_player 의 PD/toquerang ramp 는 via 이벤트 단위로 작동** — 매 스텝마다 evaluate, ramp_start_step=-1 이면 미활성. 외부에서 직접 게인 set 하지 말고 이벤트 행을 trajectory CSV에 추가하는 식으로 풀 것.

---

## 문서화 규칙

- 중간 과정에서 기록이 필요한 결정/분석/기획은 `dvcc/` 폴더에 `.md`로 추가. 추가시 prefix를 지킬것.
- prefix는 '00_제목.md' 십의 자리는 그룹묶음, 일의 자리는 그룹내 순서로
- **매 작업 시작 시 `dvcc/` 디렉토리를 먼저 확인할 것.** 기존 문서 맥락을 반영하고, 내용이 낡았으면 수정, 더 이상 유효하지 않거나 불필요하면 **파일 삭제도 허용**. `dvcc/`는 살아있는 노트 폴더로 관리.
- `README.md`는 사용자용 매뉴얼 — 직접 요청 없으면 수정 금지.

---

## 참고

- `README.md` — 사용자용 운영 매뉴얼 + Newton 설정·CSV 포맷·ROS2 토픽 매핑 표
- `dvcc/` — 프로젝트 메타 문서 (리팩토링 원칙, 커밋 컨벤션 등)
- `src/allex/config/physics_settings.py` — 물리 오버라이드 단일 진입점
- `src/allex/core/newton_bridge.py` — equality 주입 + gravcomp 적용 finalize 후크
- `tools/regen_joint_config.py` — MJCF → joint_config.json 재생성
- `data/_plot_torque_compare*.py` — 토크 비교 플롯 스크립트 (sim CSV vs rosbag)
