from diffwave.preprocess import transform, _transform_audio
import torch
import torch.nn.functional as F
import numpy as np

# torch.tensor is preferred over torch.Tensor

#@torch.no_grad()
def ls_mae(audio_true: torch.tensor, audio_pred: torch.tensor, sr: int) -> torch.tensor:
    """
    Compute the log-mel spectrogram mean absolute error (LS-MAE) between two audio signals.
    """

    assert audio_true.ndim == audio_pred.ndim, "Audio tensors must have the same number of dimensions."

    if audio_true.shape != audio_pred.shape:
        # pad the shorter audio signal with zeros
        max_len = max(audio_true.shape[-1], audio_pred.shape[-1])
        audio_true = F.pad(audio_true, (0, max_len - audio_true.shape[-1]))
        audio_pred = F.pad(audio_pred, (0, max_len - audio_pred.shape[-1]))

    # Should use same STFT as used for conditioning.
    spec_true = _transform_audio(audio_true, sr, save=False)
    spec_pred = _transform_audio(audio_pred, sr, save=False)

    return torch.mean(torch.abs(spec_true - spec_pred))


def forward_diffuse_audio(audio: torch.tensor, t: int, noise_schedule: torch.tensor) -> torch.tensor:
    """
    Generates a noisy audio signal at time step t of the forward diffusion process.
    """

    assert t < len(noise_schedule), "Time step must be less than the length of the noise schedule."

    # t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)

    beta = np.array(noise_schedule)
    noise_level = np.cumprod(1 - beta)
    noise_level = torch.tensor(noise_level.astype(np.float32))

    # noise_scale = noise_level[t].unsqueeze(1)
    noise_scale = noise_level[t]
    noise_scale_sqrt = noise_scale**0.5
    noise = torch.randn_like(audio)
    noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * noise
    return noisy_audio