"""
ALLEX Digital Twin 시뮬레이션 루프
"""


class ALLEXSimulationLoop:
    """시뮬레이션 루프 관리"""

    def __init__(self):
        self._script_generator = None
        self._is_running = False

    def set_script_generator(self, generator):
        self._script_generator = generator
        self._is_running = True

    def update(self, step: float):
        """시뮬레이션 스텝 — generator를 한 단계 진행

        Returns:
            bool: True이면 시뮬레이션 종료, False이면 계속
        """
        if self._script_generator is None:
            return True

        try:
            next(self._script_generator)
            return False
        except StopIteration:
            self._is_running = False
            return True

    def stop(self):
        self._is_running = False
        self._script_generator = None

    def is_running(self):
        return self._is_running

    def reset(self):
        self._script_generator = None
        self._is_running = False
