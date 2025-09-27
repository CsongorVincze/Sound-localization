import numpy as np
import matplotlib.pyplot as plt
# ? a program a terracsot ugy hozza letre h minden koordinata egesz lesz
# ? ez lehet h problemas ha pontos egysegekben akarunk szamolni

dt = 0.01
t = np.arange(0, 1, dt)
c = 340  # m/s hangsebesseg
f = 5  # hz frekvencia
res = 0.1  # a felbontasa a ternek
space_dims = np.array([100.0, 100.0])  # a ter merete amiben dolgozunk
coord = np.zeros(
    (int(space_dims[0] / res), int(space_dims[1] / res)), dtype=float
)  # minden pont koordinataja

s_coord = np.array(
    [3 / res, 5 / res]
)  # ? ezeket lehet at kell allitani (a source helye)


def Distance_form_source(x, y, s_coord):
    return np.sqrt(pow(x - s_coord[0], 2) + pow(y - s_coord[1], 2))


def Single_sound_source(f, c, coord, t, fi, s_coord):  # this is a sinusoidal source
    for j in range(int(space_dims[0] / res)):
        for i in range(int(space_dims[1] / res)):
            dist = Distance_form_source(i, j, s_coord)
            z = np.sin(
                2 * np.pi * f / c * dist - 2 * np.pi * f * t - fi
            )  # ? csekk h jo-e (ez adja meg a kiterest)
            coord[j, i] = z  # todo ide majd vissza kell rakni a +-t


# MIKROFONOK CUCCAI

center_mic = np.array([70 / res, 60 / res])  # a mikrofonkor kozepe
num_mics = 7  # hany mikrofonnal dolgozunk
mic_radius = 5 / res  # mekkor a kor sugara

mic_coord = np.zeros((num_mics, 2))
mic_results = np.zeros((num_mics, len(t)))
print(mic_results)


def Mic_array(
    center_mic, num_mics, mic_coord, mic_results
):  # a mikrofonok pozicionalasa
    for i in range(num_mics):
        mic_coord[i, 0] = int(
            center_mic[0] - np.sin(i * 2 * np.pi / num_mics) * mic_radius
        )
        mic_coord[i, 1] = int(
            center_mic[1] + np.cos(i * 2 * np.pi / num_mics) * mic_radius
        )


def Mic_results(mic_results, mic_coord, t_):  # a mikrofonok altal vett jelek gyujtese
    # for i in range(num_mics):
    mic_results[i, t_] = coord[int(mic_coord[i, 0]), int(mic_coord[i, 1])]
    # print(f"{mic_results[i, t_]:.2f}")


Mic_array(
    center_mic, num_mics, mic_coord, mic_results
)  # kiszamoljuk h hol vannak a mikrofonok

fig, axis = plt.subplots(2)

for j in range(len(t)):
    print(j)
    Single_sound_source(f, c, coord, t[j], 0, s_coord)
    Mic_results(mic_results, mic_coord, j)

print(mic_results)
for i in range(num_mics):
    axis[1].plot(t, mic_results[i])


im = axis[0].imshow(coord, cmap="inferno")
fig.colorbar(im, ax=axis[0])
axis[0].scatter(mic_coord[:, 0], mic_coord[:, 1], c="#79E614", marker="x")

plt.legend()
plt.show()
