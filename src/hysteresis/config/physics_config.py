"""Hysteresis world/physics knobs (독립).

Isaac Sim `World` 생성 시 LoadButton 에 넘기는 값들. ALLEX 본체와
독립적으로 튜닝한다 — Hysteresis 는 실측 이력곡선 재현이 목표라
더 촘촘한 physics_dt 로 가기 쉬우므로 여기서 바로 조절한다.

MJCF equality softness(`solref`/`solimp`) 는 config/joint_config.json 의
top-level 필드에서 관리된다. 이 파일은 dt / gravity 만 담는다.
"""


class PhysicsConfig:
    """Hysteresis 시나리오 world/physics 설정."""

    # MuJoCo equality constraint 의 solref timeconst 하한은 2 * PHYSICS_DT 이다.
    # dt=1/200 → 하한 0.010 s. 더 빡빡한 coupling 이 필요하면 여기부터 줄인다.
    PHYSICS_DT = 1 / 200.0
    RENDERING_DT = 1 / 30.0

    GRAVITY = [0.0, 0.0, -9.81]
