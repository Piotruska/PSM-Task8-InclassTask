import numpy as np
import matplotlib.pyplot as plt


def f(t, y, A, B, C):
    x, y, z = y
    dxdt = A * y - A * x
    dydt = -x * z + B * x - y
    dzdt = x * y - C * z
    return np.array([dxdt, dydt, dzdt])


def euler_method(f, y0, t0, dt, tmax, *args):
    y = [np.array(y0)]
    t = t0
    while t < tmax:
        y_next = y[-1] + dt * f(t, y[-1], *args)
        y.append(y_next)
        t += dt
    return np.array(y)


def midpoint_method(f, y0, t0, dt, tmax, *args):
    y = [np.array(y0)]
    t = t0
    while t < tmax:
        k1 = dt * f(t, y[-1], *args)
        k2 = dt * f(t + 0.5 * dt, y[-1] + 0.5 * k1, *args)
        y_next = y[-1] + k2
        y.append(y_next)
        t += dt
    return np.array(y)


def rk4_method(f, y0, t0, dt, tmax, *args):
    y = [np.array(y0)]
    t = t0
    while t < tmax:
        k1 = dt * f(t, y[-1], *args)
        k2 = dt * f(t + 0.5 * dt, y[-1] + 0.5 * k1, *args)
        k3 = dt * f(t + 0.5 * dt, y[-1] + 0.5 * k2, *args)
        k4 = dt * f(t + dt, y[-1] + k3, *args)
        y_next = y[-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        y.append(y_next)
        t += dt
    return np.array(y)


y0 = [1, 1, 1]
A = 10
B = 25
C = 8 / 3
t0 = 0
dt = 0.02
tmax = 70


results_euler = euler_method(f, y0, t0, dt, tmax, A, B, C)
results_midpoint = midpoint_method(f, y0, t0, dt, tmax, A, B, C)
results_rk4 = rk4_method(f, y0, t0, dt, tmax, A, B, C)

x_euler = results_euler[:, 0]
z_euler = results_euler[:, 2]
x_midpoint = results_midpoint[:, 0]
z_midpoint = results_midpoint[:, 2]
x_rk4 = results_rk4[:, 0]
z_rk4 = results_rk4[:, 2]

plt.figure()
plt.plot(x_euler, z_euler, linewidth=0.7)
plt.xlabel('x')
plt.ylabel('z')
plt.title('Euler Method')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(x_midpoint, z_midpoint, linewidth=0.7)
plt.xlabel('x')
plt.ylabel('z')
plt.title('Midpoint Method')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(x_rk4, z_rk4, linewidth=0.7)
plt.xlabel('x')
plt.ylabel('z')
plt.title('RK4 Method')
plt.grid(True)
plt.show()
