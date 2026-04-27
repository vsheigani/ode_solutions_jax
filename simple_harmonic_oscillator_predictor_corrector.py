import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def deriv(y):
    """Derivatives for simple harmonic oscillator: x'=v, v'=-x"""
    return jnp.array([y[1], -y[0]])


@jax.jit
def step_predictor_corrector(y, step):
    """One predictor-corrector (Euler-Trapezoidal) step."""
    f = deriv(y)
    yc = y + step * f          # predictor (Euler)
    fc = deriv(yc)
    y_new = y + 0.5 * step * (f + fc)  # corrector (trapezoidal)
    return y_new


def run_simulation():
    xinit = 0.0
    vinit = 5.0
    step = 0.01
    t_end = 10.0

    y = jnp.array([xinit, vinit])
    time = 0.0
    energy = 0.5 * xinit ** 2 + 0.5 * vinit ** 2
    exact = 0.0

    times, positions, velocities, errors, energies = [], [], [], [], []

    while time < t_end:
        times.append(time)
        positions.append(float(y[0]))
        velocities.append(float(y[1]))
        errors.append(abs(exact - float(y[0])))
        energies.append(float(energy))

        y = step_predictor_corrector(y, step)
        energy = 0.5 * y[0] ** 2 + 0.5 * y[1] ** 2
        time += step
        exact = 5.0 * jnp.sin(time)

    return {
        "time": times,
        "position": positions,
        "velocity": velocities,
        "error": errors,
        "energy": energies,
    }


if __name__ == "__main__":
    run_simulation()
