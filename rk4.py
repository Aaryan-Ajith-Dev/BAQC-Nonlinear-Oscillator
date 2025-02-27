import numpy as np

def rk4(func_list, x_list, t, dt = 0.001, n = 500):
    x_list = [x_list]
    def single_step_rk4(func_list, x_list, t, dt):
        k1 = [f(*x_list, t) for f in func_list]
        k2 = [f(*(x + 0.5 * dt * k for x, k in zip(x_list, k1)), t + 0.5 * dt) for f in func_list]
        k3 = [f(*(x + 0.5 * dt * k for x, k in zip(x_list, k2)), t + 0.5 * dt) for f in func_list]
        k4 = [f(*(x + dt * k for x, k in zip(x_list, k3)), t + dt) for f in func_list]
        return [x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4) for x, k1, k2, k3, k4 in zip(x_list, k1, k2, k3, k4)]
    for _ in range(n - 1):
        x_list.append(single_step_rk4(func_list, x_list[-1], t, dt))
        t += dt
    # convert rows in x_list to cols
    x_list = np.array(x_list).T
    return x_list, np.arange(0, n * dt, dt) 