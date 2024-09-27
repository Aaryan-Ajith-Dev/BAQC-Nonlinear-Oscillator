# what about frequency? also change to RK4 -- use lorenz attractor plot

import numpy as np
import matplotlib.pyplot as plt

# p(0) = 0, p(500) = 5

K = 1
DELTA = 1
P_init = 0
P_final = 20*K
P_tfinal = 500/K


def p(t):
    return (P_final - P_init) * t / (P_tfinal)


def getPath(x10,x20,time_step,n):
    def f1(x,y,t):    
        return y*(DELTA + p(t) + K*(x**2 + y**2))

    def f2(x,y,t):
        return x*(-DELTA + p(t) - K*(x**2 + y**2))
    
    x1=[x10]
    x2=[x20]
    t=[0]       
    dt=time_step    
    for _ in range(n):
        d_x1 = f1(x1[-1],x2[-1],t[-1]) * dt
        x1.append(x1[-1] + d_x1)
        d_x2 = f2(x1[-1],x2[-1],t[-1]) * dt
        x2.append(x2[-1] + d_x2)
        t.append(t[-1] + dt)
    return x1, x2, t

def plot_bifurcation(x_init=0.1, y_init=0.1, x_lim=20,y_lim=5):
    dt = 0.001
    n = int(P_tfinal / dt)
    x,y,t = getPath(x_init, y_init, dt, n)
    p_delta = [p(i)/DELTA for i in t]

    _, axis = plt.subplots(2)

    ax = axis.flat
    [a.label_outer() for a in ax]
    ax[0].set(xlabel='p / delta', ylabel='x')
    ax[1].set(xlabel='p / delta', ylabel='y')

    axis[0].plot(p_delta, x)

    plt.xlim(0, x_lim)
    plt.ylim(-y_lim, y_lim)

    axis[1].plot(p_delta, y)

    plt.xlim(0, x_lim)
    plt.ylim(-y_lim, y_lim)


if __name__ == '__main__':
    plot_bifurcation(0.01, 0.01)
    plot_bifurcation(0.1, 0.1)
    plt.show()
