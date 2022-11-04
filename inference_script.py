import torchaudio
import torch
data_dir = '/Users/lukgar/Desktop/exjobb/datasets/LJSpeech-1.1/wavs'
model_dir = 'src/diffwave-ljspeech-22kHz-1000578.pt'

filename = data_dir + '/LJ001-0004.wav'
from diffwave.preprocess import transform

spec = transform(filename, save=False)
spectrogram = spec.view(1, *spec.shape)

from diffwave.inference import predict
audio, sr = predict(spectrogram, model_dir, device=torch.device('cpu'), fast_sampling=True)