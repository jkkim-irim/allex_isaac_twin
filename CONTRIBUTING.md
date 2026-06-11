# Contributing to ALLEX IsaacTwin

ALLEX IsaacTwin 에 기여해주셔서 감사합니다. 이 문서는 PR/이슈를 올리기 전에 한 번 훑어보면 좋은 가이드입니다.

## 개발 환경

- conda env `isaacsim` (Python 3.12) + Isaac Sim 6.0.0 — 자세한 설치는 [README.md](README.md#설치) 참조
- Git LFS 필수 — USD/STL 자산이 LFS 로 관리됨

```bash
sudo apt install git-lfs   # 또는 brew install git-lfs
git lfs install
git clone <repo-url> && cd allex_isaac_twin
git lfs pull
```

## 변경 사항을 보내기 전 체크리스트

- [ ] Isaac Sim 에서 LOAD → RUN 정상 동작 확인
- [ ] `trajectory/demo1_dynamic_group` 한 번 재생 후 진동/스냅 없음
- [ ] 새 파일은 docstring 1줄 이상, public 함수는 args/returns 명시
- [ ] 기존 동작 깨졌으면 README/CHANGELOG 갱신

## 커밋 메시지

[Conventional Commits](https://www.conventionalcommits.org/) 한국어 변형. 메시지만 보고 변경 종류와 범위 즉시 파악 가능하게:

```
<type>[(scope)]: <description>

[body]
```

| 타입 | 의미 | 예시 |
|---|---|---|
| `feat` | 기능 추가 | `feat(motor): add per-step ramp` |
| `fix` | 버그 수정 | `fix(traj): resolve K_vel event misorder` |
| `refactor` | 리팩터링 (동작 변경 없음) | `refactor(scenario): extract motor mirror lifecycle` |
| `docs` | 문서 | `docs(readme): add Quickstart` |
| `chore` | 빌드/패키지/git | `chore(lfs): track *.usd` |
| `data` | 데이터/궤적 | `data(traj): add gravcomp_test_group` |
| `test` | 테스트 | `test(jk_kernel): verify polynomial accuracy` |

규칙:
- 제목 50자 이내, 첫 글자 소문자, 마침표 없음
- 명령문 사용 (Add 가 아니라 add — 한국어는 명사형도 OK)
- 본문은 **무엇을 / 왜** 위주 (어떻게는 코드에)

## 코드 스타일

`dvcc/01_refactoring_guide.md` 의 원칙을 따릅니다:

- **YAGNI**: 미래 대비 미리 만들기 금지
- **KISS**: 설명이 필요한 구조는 실패한 설계
- **선 병합, 후 분리**: 800줄 이상이거나 명확한 재사용 포인트가 보일 때만 분리
- **Dead code 즉시 삭제**
- **불필요한 예외처리 금지** — 시스템 경계에서만 최소 예외처리. 에러 삼키면 디버깅 불가

## 변경 가능한 부분

- `src/allex/config/*.json` — 런타임 튜닝 OK
- `src/allex/ui.py` — 새 패널 추가 시 동일 파일 안에 클래스 추가 (`AllExUI.build_ui()` 끝부분에서 wire-up)
- `trajectory/<group>/*.csv` — 새 시나리오 자유 추가
- `tools/*.py` — 새 유틸리티 자유 추가

## 변경 시 사용자 확인이 필요한 부분 (불변식)

`README.md` 의 핵심 기능 섹션 + `src/allex/config/physics_config.json::ramp_step_sizes`
의 `_doc` 같은 내부 invariant 설명. 특히:

- `pd_scale = π/180` (Newton USD importer 의 deg→rad 보정)
- `use_cuda_graph = True` (그래프 재기록 방지)
- `mjc:gravcomp` 인증 body 1개 이상 빌드 타임 필수 (`m.ngravcomp > 0` gate)
- `active_joints` 순서 (trajectory CSV 의 joint_1..N 인덱스와 직결)

이 항목들 변경은 PR 본문에 변경 이유를 명시해주세요.

## 라이선스

이 프로젝트는 [LICENSE](LICENSE) 의 약관을 따릅니다. PR 제출 시 동일 라이선스에 동의하는 것으로 간주합니다.
