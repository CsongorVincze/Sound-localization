import numpy as np
from numpy import random
import matplotlib.pyplot as plt

t_start = 0
t_end = 100
dt = 0.01
f_max = 5  # a freki amivel "feltekerjuk a jelet"
df = 0.01

t = np.arange(t_start, t_end, dt)


def g(t: np.ndarray) -> np.ndarray:  # itt csinalunk egy osszekutyult input signal-t
    print("these are the frequencies of the sin-s:")
    num_sines = 5
    g_collect = np.zeros_like(t, dtype=float)
    for _ in range(num_sines):
        r = random.rand(3)
        g_ = r[0] * np.sin(r[1] * t + r[2])  # egy sin random parameterekkel
        g_collect += g_  # szepen osszeadogatjuk a komponenseket
        print(f"{r[1]:.2f}")
    return g_collect


def Integrator(fun: np.ndarray, x: np.ndarray):  # bekerjuk a fuggvenyt es az idotombot
    dx = x[1] - x[0]  # ?remelem h itt nem lesz pontatlan
    area_accum = 0
    i = 0
    for _ in x:
        area_accum += fun[i] * dx
        i += 1

    return area_accum


def Fourier(g: np.ndarray, t: np.ndarray, f, df) -> np.ndarray:
    f_arr = np.arange(0, f, df)
    result = np.zeros(f_arr.size)
    i = 0
    for _ in f_arr:
        mul_1 = g * np.cos(f_arr[i] * t)  # ?
        mul_2 = g * np.sin(f_arr[i] * t)
        res_1 = Integrator(mul_1, t)
        res_2 = Integrator(mul_2, t)
        result[i] = np.sqrt(res_1**2 + res_2**2)

        i += 1

    return result


g__ = g(t)
G = Fourier(g__, t, f_max, df)


plt.figure(figsize=(9, 2))

plt.subplot(121)
plt.plot(t, g__, color="blue")
plt.title("Signal")
plt.xlabel("time (s)")

plt.subplot(122)
f_arr = np.arange(0, f_max, df)
plt.plot(f_arr, G, color="red")
plt.title("Fourier transform of the signal")
plt.xlabel("frequency (Hz)")  #! lehet h itt van egy 2pi-s elcsuszas a freki miatt

plt.show()
