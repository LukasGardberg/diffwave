# Load pretrained model and look at the spectrogram cnn's weights
import torch
import torchaudio
import os
from diffwave.params import AttrDict, params as base_params
from diffwave.model import DiffWave

import utils
from diffwave.inference import predict
from diffwave.preprocess import transform, _transform_audio
import matplotlib.pyplot as plt

device = torch.device('cpu')

model_dir = 'diffwave-ljspeech-22kHz-1000578.pt'

checkpoint = torch.load(model_dir, map_location=device)
model = DiffWave(AttrDict(base_params)).to(device)

data_dir = '/Users/lukgar/Desktop/exjobb/datasets/LJSpeech-1.1/wavs'
filename = data_dir + '/LJ001-0001.wav'

spec = transform(filename, save=False)
spectrogram = spec.view(1, *spec.shape)

audio, sr = torchaudio.load(filename)

# try 0 loss
loss_0 = utils.ls_mae(audio, audio, sr=sr)

# TODO: run this to make sure loss is correct
# Try to run for a handful of audio samples with more noise added to them
# to confirm proxy to convergence

noise_schedule = base_params.noise_schedule
losses = []

for i, noise in enumerate(noise_schedule):
    # get noised audio for step i
    noised_audio = utils.forward_diffuse_audio(audio, i, noise_schedule)
    losses.append(utils.ls_mae(audio, noised_audio, sr=sr).item())

print(losses[::-1])
plt.figure()
plt.plot(losses[::-1])
plt.show()