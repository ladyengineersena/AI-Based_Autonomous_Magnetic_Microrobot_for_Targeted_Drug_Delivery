from environment.physics import (
    N_DISCRETE_ACTIONS,
    RobotState,
    action_to_field_angle,
    integrate_step,
    magnetic_field_vector,
    wrap_angle,
)


def test_wrap_angle():
    assert abs(wrap_angle(3.2) + 3.083185307179586) < 1e-6 or abs(wrap_angle(3.2)) <= 3.1416
    assert abs(wrap_angle(0.0)) < 1e-9
    assert abs(abs(wrap_angle(3.141592653589793)) - 3.141592653589793) < 1e-6


def test_discrete_actions_cover_full_circle():
    angles = [action_to_field_angle(i) for i in range(N_DISCRETE_ACTIONS)]
    assert len(set(round(a, 8) for a in angles)) == N_DISCRETE_ACTIONS
    assert abs(angles[0]) < 1e-9
    assert abs(angles[2] - 1.5707963267948966) < 1e-9


def test_magnetic_field_unit_vector():
    field = magnetic_field_vector(0.0, 1.0)
    assert abs(field[0] - 1.0) < 1e-9
    assert abs(field[1]) < 1e-9


def test_integrate_moves_along_field():
    state = RobotState(x=0.5, y=0.5, theta=0.0)
    nxt = integrate_step(state, field_angle=0.0, dt=0.1, v_max=1.0, k_align=8.0, max_omega=6.0)
    assert nxt.x > state.x
    assert abs(nxt.y - state.y) < 1e-9
