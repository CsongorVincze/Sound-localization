import numpy as np

SPEED_OF_SOUND = 343.0
MIC_SPACING = 0.0465
#? ezeket biztos h jo adatok

MIC_ANGLE_DEG = np.array([45, 315, 225, 135]) #? ez itt nem tunik jonak
MIC_RADIUS = MIC_SPACING /np.sqrt(2)

MIC_POSITIONS = np.array([
    [MIC_RADIUS * np.sin(np.radians(MIC_ANGLE_DEG[i])),
     MIC_RADIUS * np.cos(np.radians(MIC_ANGLE_DEG[i]))]
    for i in range(4)
])
#todo itt a mic adatok eleg erdekesek, at kene nezni

def gcc_