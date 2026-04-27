import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def f(y1):
    return y1


@jax.jit
def step_leapfrog(y0, y1, h):
    """One leapfrog step: y2 = y0 + 2*h*f(y1)"""
    return y0 + 2.0 * h * f(y1)


def run_simulation():
    h = 0.1
    time = 0.1

    y0 = jnp.exp(time - h)
    y1 = jnp.exp(time)

    times, leapfrog_vals, exact_vals = [], [], []

    while time <= 2.0:
        y2 = step_leapfrog(y0, y1, h)
        exact = jnp.exp(time + h)

        times.append(float(time))
        leapfrog_vals.append(float(y2))
        exact_vals.append(float(exact))

        y0 = y1
        y1 = y2
        time = round(time + h, 10)

    return {
        "time": times,
        "leapfrog": leapfrog_vals,
        "exact": exact_vals,
    }


if __name__ == "__main__":
    run_simulation()
