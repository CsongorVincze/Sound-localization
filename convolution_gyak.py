import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

t = np.arange(-10, 10, 0.01)  # kis idodarabkak

# sinos hullam parameterei
f0 = 1
A_s = 3
phase_s_0 = 1


def sin_fn(t, phase):
    s = np.where(t >= 0, A_s + A_s * np.sin(f0 * t + phase), 0.0)
    return s


s = sin_fn(t, phase_s_0)
plt.subplot(211)
(S,) = plt.plot(t, s)

# kernel fuggveny (ez egy linearis fuggveny 2 hatar kozott)
A_k = 2
phase_k_0 = 0


def kernel_fn(t, phase):
    blabla = np.where((t + phase >= 2) & (t + phase <= 5), A_k * (t + phase), 0)
    return blabla


k = kernel_fn(t, phase_k_0)
(K,) = plt.plot(t, k)  # ez az a plot amit valtoztatni fogunk
(K_original,) = plt.plot(t, k)

phase_ = np.arange(0, 10, 0.01)
mul_ = np.zeros(phase_.size)
plt.subplot(212)
(new,) = plt.plot(phase_, mul_)
plt.ylim(0, 15)
# plt.title("Convolution of the two distribution")

ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
phase_slider = Slider(ax_slider, "Sum", -8.0, 12.0, valinit=phase_k_0)


def update(val):
    phase = phase_slider.val
    nana = kernel_fn(-t, phase)
    K.set_ydata(nana)
    fig.canvas.draw_idle()

    # print(k[np.arange(0, t.size, 10)]) # teszt hogy mikor erzekeli a slider valtoztatasat
    mul = np.sum(s * nana) / 1000  # todo itt a normalast ki kell talalni
    print(mul)
    # megadja, hogy egy bizonyos osszeg korul mekkora valoszinuseggel lesz a ket valtozo összege

    mul_[int(phase / phase_[-1] * phase_.size)] = mul

    new.set_ydata(mul_)


phase_slider.on_changed(update)


plt.show()

plt.plot(phase_, mul_)
plt.show()
