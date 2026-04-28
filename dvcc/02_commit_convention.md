# 커밋 컨벤션

Conventional Commits 기반. 메시지만 보고 변경 종류와 범위를 즉시 파악 가능하게. 메시지는 한글로

## 구조

```
<type>[(scope)]: <description>

[body]
```

## 타입

| 타입 | 의미 | 예시 |
|------|------|------|
| `feat` | 기능 추가 | `feat(dexblind): add hammer shape randomization` |
| `fix` | 버그 수정 | `fix(newton): resolve solver convergence failure` |
| `refactor` | 리팩터링 | `refactor: optimize observation computation` |
| `docs` | 문서 | `docs: update sim2real pipeline spec` |
| `chore` | 빌드/패키지 | `chore: update rsl_rl dependency` |
| `test` | 테스트 | `test: add reward function unit test` |
| `style` | 포맷팅만 | `style: apply black formatter` |

## 규칙

- 제목 50자 이내, 첫 글자 소문자, 마침표 없음
- 명령문 사용 (Add, Fix — Added, Fixed 아님)
- 본문은 **무엇을/왜** 위주 (어떻게는 코드에)
- scope 예시: `newton`, `physx`, `sim2real`, `dexblind`, `assets`

## 에이전트 커밋+push 절차

1. `git status` + `git diff --stat`으로 변경 파악
2. 위 규칙에 맞게 메시지 작성
3. 관련 파일만 `git add`
4. `git commit` → `git log -1 --oneline`으로 검증
5. push 요청 시: `git push` (추적 없으면 `git push -u origin <branch>`)

## fork 운용

```bash
# origin = 본인 fork, upstream = 원본
git remote rename origin upstream
git remote add origin git@github.com:<user>/IsaacLab.git
git push -u origin <branch>
```
