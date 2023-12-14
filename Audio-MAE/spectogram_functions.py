##########################################
# Add some spectogram functions
##########################################

import torchaudio
import torch
from torchaudio.compliance import kaldi
import matplotlib.pyplot as plt

#MELBINS=128  # 64 for ESC, 128 for AudioSet (Copilot comment)
#TARGET_LEN=1024  # 512 for ESC, 1024 for AudioSet (Copilot comment)
def wav2fbank(filename, MELBINS = 128, TARGET_LEN = 1024):
    waveform, sr = torchaudio.load(filename)
    waveform = waveform - waveform.mean()
    # 498 128
    fbank = kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, 
                        window_type='hanning', num_mel_bins=MELBINS, dither=0.0, frame_shift=10)
    # AudioSet: 1024 (16K sr)
    # ESC: 512 (8K sr)
    n_frames = fbank.shape[0]
    p = TARGET_LEN - n_frames
    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:TARGET_LEN, :]
    return fbank

def norm_fbank(fbank):
    norm_mean= -4.2677393
    norm_std= 4.5689974
    fbank = (fbank - norm_mean) / (norm_std * 2)
    return fbank

def display_fbank(bank, minmin=None, maxmax=None):
    plt.figure(figsize=(20, 4))
    plt.imshow(20*bank.T.numpy(), origin='lower', interpolation='nearest', vmax=maxmax, vmin=minmin,  aspect='auto')