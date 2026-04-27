# ODE Numerical Methods

This project implements and compares three classical numerical integrators for ordinary differential equations (ODEs). Each method is implemented in both Python (using JAX for JIT compilation) and C++, and applied to a different test problem chosen to highlight the method's strengths and limitations.

| Method | Equation | Exact solution |
|---|---|---|
| Predictor-Corrector (Euler-Trapezoidal) | $\ddot{x} = -x$ | $x(t) = 5\sin t$ |
| Leapfrog (Störmer–Verlet) | $y' = y$ | $y(t) = e^t$ |
| Runge-Kutta 4 (RK4) | $y' = -y$ | $y(t) = e^{-t}$ |

Results and plots are in `results.ipynb`.

---

## Setup and Usage

### Prerequisites

- [uv](https://docs.astral.sh/uv/) — Python package and project manager
- Python 3.12+

### Install dependencies

```bash
uv sync
```

This creates a virtual environment and installs all dependencies (`jax`, `matplotlib`, `ipykernel`, `ipython`).

### Run the notebook

```bash
uv run jupyter notebook results.ipynb
```

Or open it in VS Code with the Jupyter extension — select the `.venv` kernel created by `uv`.

### Run the Python scripts directly

```bash
uv run python simple_harmonic_oscillator_predictor_corrector.py
uv run python leap_frog_method.py
uv run python simple_ode_RK4.py
```

### Build and run the C++ implementations

```bash
g++ -O2 -o sho simple_harmonic_oscillator_predictor_corrector.cpp && ./sho
g++ -O2 -o leapfrog leap_frog_method.cpp && ./leapfrog
g++ -O2 -o rk4 simple_ode_RK4.cpp && ./rk4
```

---

## Predictor-Corrector (Euler-Trapezoidal)

### Problem

The simple harmonic oscillator $\ddot{x} = -x$ is rewritten as a first-order system:

$$\dot{x} = v, \qquad \dot{v} = -x$$

with $x(0) = 0$, $v(0) = 5$, giving exact solution $x(t) = 5\sin t$.

### Method

Given $(x_n, v_n)$, each step proceeds in two stages:

**Predict** (explicit Euler):
$$x^* = x_n + h\,v_n, \qquad v^* = v_n - h\,x_n$$

**Correct** (trapezoidal rule):
$$x_{n+1} = x_n + \frac{h}{2}(v_n + v^*), \qquad v_{n+1} = v_n - \frac{h}{2}(x_n + x^*)$$

### Properties

- **2nd-order accurate**: global error $\mathcal{O}(h^2)$.
- **Not symplectic**: the total energy $E = \frac{1}{2}(x^2 + v^2)$ drifts over long integrations. A spiral in the phase portrait ($v$ vs $x$) reveals this drift — a perfect integrator traces a closed circle.

---

## Leapfrog (Störmer–Verlet)

### Problem

Scalar ODE $y' = y$ with exact solution $y(t) = e^t$.

### Method

The leapfrog advances the solution using a half-step:

$$y_{n+1/2} = y_n + \frac{h}{2}\,f(y_n), \qquad y_{n+1} = y_{n+1/2} + \frac{h}{2}\,f(y_{n+1/2})$$

### Properties

- **2nd-order accurate**: global error $\mathcal{O}(h^2)$.
- **Symplectic and time-reversible**: conserves a modified energy exactly, so there is no secular drift — ideal for long-time Hamiltonian integration.

---

## Runge-Kutta 4 (RK4)

### Problem

Scalar ODE $y' = -y$ on $[0.5,\, 0.7]$ with $y(0.5) = e^{-0.5}$, exact solution $y(t) = e^{-t}$.

### Method

Each step computes four slope estimates:

$$k_1 = h\,f(y_n), \quad k_2 = h\,f\!\left(y_n + \tfrac{k_1}{2}\right), \quad k_3 = h\,f\!\left(y_n + \tfrac{k_2}{2}\right), \quad k_4 = h\,f(y_n + k_3)$$

and advances with the weighted average:

$$y_{n+1} = y_n + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

### Properties

- **4th-order accurate**: local truncation error $\mathcal{O}(h^5)$, global error $\mathcal{O}(h^4)$.
- **Not symplectic**: unsuitable for very long Hamiltonian integrations, but highly accurate for short-to-medium intervals.

### Parts

- **(a)** Integrate with $h_1 = 0.2$.
- **(b)** Integrate with $h_2 = 0.1$ (halved step) and compare accuracy.
- **(c, d)** Compute an adaptive step size $h_0$ via a Richardson-style estimate. Since RK4 error scales as $h^4$, the step needed to achieve tolerance $\varepsilon = 10^{-7}$ is:

$$h_0 = h_1 \left|\frac{\varepsilon}{y_{h_2} - y_{h_1}}\right|$$

  Part (d) confirms $h_0$ by verifying the RK4 value at $t = 0.5 + h_0$ agrees with the exact solution to within $\varepsilon$.
