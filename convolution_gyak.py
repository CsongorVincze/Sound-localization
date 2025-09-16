import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

t = np.arange(0, 10, 0.01)
f0 = 3
s = 3 * np.sin(f0 * t)
(l,) = plt.plot(t, s)

ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
fq_slider = Slider(ax_slider, "Fq", 0.0, 10.0, valinit=3.0)


def update(val):
    fq = fq_slider.val
    l.set_ydata(3 * np.sin(fq * t))
    fig.canvas.draw_idle()


fq_slider.on_changed(update)

plt.show()
