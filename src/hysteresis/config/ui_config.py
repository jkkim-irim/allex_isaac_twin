"""Hysteresis UI-level knobs (독립).

Hysteresis 예제의 UI/lifecycle 튜닝 값만 담는다. 색상/레이아웃(UIColors,
UILayout) 은 프로젝트 전역 테마를 유지하기 위해
`allex.utils.ui_settings_utils` 를 계속 공유하며, 여기서는 시나리오 동작에
직접 영향을 주는 값만 둔다.
"""


class UIConfig:
    """Hysteresis 시나리오 UI 설정."""

    # RUN 이후 Articulation.initialize() 재시도 한도/대기 스텝.
    # 너무 작으면 빈 stage 상태에서 init 이 실패할 수 있다.
    MAX_INITIALIZATION_ATTEMPTS = 5
    INITIALIZATION_DELAY_STEPS = 3
