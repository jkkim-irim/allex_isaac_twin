# 리팩토링 가이드

## 핵심 원칙

- **YAGNI**: 미래를 대비한 미리 만들기 금지. 지금 필요한 것만 구현.
- **KISS**: 설명이 필요한 구조는 실패한 설계. 직관적으로.
- **선 병합, 후 분리**: 800줄 이상이거나 명확한 재사용 포인트가 보일 때만 분리.
- **Dead Code 즉시 삭제**: 미사용 import, 주석 처리된 코드, 쓰지 않는 변수.
- **불필요한 예외처리 금지**: try/except, fallback, 방어적 validation 남발 금지. 에러가 삼켜지면 디버깅이 불가능해진다. 예외처리는 시스템 경계(사용자 입력, 외부 API)에서만 최소한으로.

## 명명 규칙

- 모호한 이름 금지: `data`, `handle`, `temp` → `buffer_index`, `validate_sensor_input`
- 함수/변수명만으로 역할(What)과 이유(Why)가 드러나야 함

## 리팩토링 프로세스

1. 현상 파악 — 중복/복잡한 조건문/성능 저하 지점 식별
2. 검증 수단 확보 — 리팩토링 전후 동일 결과 증명할 환경 확인
3. 최소 단위 실행 — 함수/클래스 단위로 한 블록씩
4. 중간 검증 — 수정 직후 실행하여 사이드 이펙트 확인
5. 커밋 — `refactor:` 또는 `perf:` 접두사 사용

## 적용 범위

| 프로젝트         | 대상 경로                                                                 |
| ------------ | --------------------------------------------------------------------- |
| allex_newton | `allex_newton/source/allex_rl_dexblind/`, `allex_newton/scripts/`     |
| allex_physx  | `allex_physx/source/allex_rl_dexblind_physx/`, `allex_physx/scripts/` |

**성공 기준**: train/play 정상 실행, observation/reward/termination 동작 유지

## 체크리스트

- [ ] 가독성: 코드만 보고 1분 내 로직 파악 가능한가?
- [ ] 중복: 유사 로직 3곳 이상 반복 없는가?
- [ ] 안정성: 기존 기능 정상 동작하는가?
- [ ] config/상수 적절히 분리되었는가?
- [ ] 미사용 코드 전부 제거되었는가?
