import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def f(y):
    return -y


@jax.jit
def rk4_step(y, h):
    c1 = h * f(y)
    c2 = h * f(y + c1 / 2.0)
    c3 = h * f(y + c2 / 2.0)
    c4 = h * f(y + c3)
    return y + (c1 + 2.0 * c2 + 2.0 * c3 + c4) / 6.0


def _run_segment(t_start, t_end, h, y_init):
    times, rk4_vals, exact_vals = [], [], []
    time = t_start
    y = y_init
    while time <= t_end:
        y = rk4_step(y, h)
        exact = jnp.exp(-time)
        times.append(float(time))
        rk4_vals.append(float(y))
        exact_vals.append(float(exact))
        time = round(time + h, 10)
    return times, rk4_vals, exact_vals, float(y)


def run_simulation():
    t_start = 0.5
    t_end = 0.7
    xinit = jnp.exp(-t_start)
    h1, h2 = 0.2, 0.1

    # Part (a): h1 = 0.2
    t_a, y_a, e_a, yh1 = _run_segment(t_start, t_end, h1, xinit)

    # Part (b): h2 = 0.1
    t_b, y_b, e_b, yh2 = _run_segment(t_start, t_end, h2, xinit)

    # Part (c): adaptive h0 from Richardson-style step-size estimate
    h0 = h1 * abs(1e-7 / (yh2 - yh1))
    t_c, y_c, e_c, _ = _run_segment(t_start, t_start + h0, h0, xinit)

    return {
        "part_a": {"time": t_a, "rk4": y_a, "exact": e_a, "h": h1},
        "part_b": {"time": t_b, "rk4": y_b, "exact": e_b, "h": h2},
        "part_c": {"time": t_c, "rk4": y_c, "exact": e_c, "h": h0},
        "h0": h0,
    }


if __name__ == "__main__":
    run_simulation()
