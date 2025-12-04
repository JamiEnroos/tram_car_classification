# Main
import librosa as lb

import numpy as np
import soundfile as sf


def load_audio(path, sr=22050, duration=6.0, mono=True):
    """
    Load audio, resample to sr, force mono if desired, and pad/trim to `duration` seconds.
    Returns: y (np.ndarray, shape [samples]), sr
    """
    y, sr = lb.load(path, sr=sr, mono=mono)
    target_len = int(duration * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]
    return y, sr

def stats(x, prefix):
    stats = {}
    x = np.atleast_2d(x)

    for i in range(x.shape[0]):
        v = x[i]
        stats[f"{prefix}{i}_mean"] = float(np.mean(v))
        stats[f"{prefix}{i}_std"]  = float(np.std(v))
        """
        stats[f"{prefix}{i}_min"]  = float(np.min(v))
        stats[f"{prefix}{i}_max"]  = float(np.max(v))"""
    return stats

def get_time_domain_features(path, sr, duration=6.0):
    y, sr = load_audio(path, sr=sr, duration=duration, mono=True)

    features = {}
    # Time domain
    zcr_mean = np.mean(lb.feature.zero_crossings(y)[0])
    zcr_mean = np.mean(zcr)
    features.update(stats(zcr, "zcr_"))
    rms = lb.feature.rms(y=y, sr=sr)[0]
    features.update(stats(rms, "rms_"))

    # Frequency domain
    # magnitude STFT
    stft = np.abs(lb.stft(y, n_fft=sr, hop_length=sr//2))

    # Features
    spec_centroid = lb.feature.spectral_centroid(S=stft, sr=sr)[0]
    spec_bw = lb.feature.spectral_bandwidth(S=stft, sr=sr)[0]
    spec_rolloff = lb.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.85)[0]

    features.update(stats(spec_centroid, "spec_centroid_"))
    features.update(stats(spec_bw, "spec_bw_"))
    features.update(stats(spec_rolloff, "spec_rolloff_"))

    mfcc = lb.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.update(stats(mfcc, "mfcc"))

