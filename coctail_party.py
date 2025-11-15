import soundfile as sf
import numpy as np
import torch


class source:
    def __init__(self, pos, wave, sr):
        self.pos = pos
        self.wave = wave
        self.sr = sr
        
        
class mic_array:
    def __init__(self, center, radius, num_mics, wave):
        self.center = center
        self.radius = radius
        self.num_mics = num_mics
        self.wave = wave
        
        
        
    def array_geo(self):
        self.pos = torch.zeros(self.num_mics, 2)
        for i in self.num_mics:
            self.pos[i, 0] = self.center[i, 0] + self.radius * torch.cos(i * 2*torch.pi/self.num_mics)
            self.pos[i, 1] = self.center[i, 1] + self.radius * torch.sin(i * 2*torch.pi/self.num_mics)
            
        return self.pos
        
        
        


def dist(a, b):  # tavolsagok kiszamitasara
    return torch.sqrt(torch.pow((b[1] - a[1]), 2) + torch.pow((b[0] - a[0]), 2))


dum1, sr1 = sf.read("Duma_1.flac")
dum2, sr2 = sf.read("Duma_2.flac")

dum1 = torch.from_numpy(dum1)
dum2 = torch.from_numpy(dum2)


duma1 = source(torch.tensor([2, 4]), dum1, sr1)
duma2 = source(torch.tensor([6, 9]), dum2, sr2)

print(duma1.wave.shape)

mavolsag = dist(duma1.pos, duma2.pos)

print(mavolsag)
