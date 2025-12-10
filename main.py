# Main
import os

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

def get_stats(x, features):
    mean = np.mean(x)
    std = np.std(x)
    return np.array([mean, std])

def get_time_domain_features(path, sr, duration=6.0):
    y, sr = load_audio(path, sr=sr, duration=duration, mono=True)

    time_domain_features = np.zeros(0)
    # Time domain
    zcr = lb.feature.zero_crossings(y)[0]
    zcr_stats = get_stats(zcr, time_domain_features)

    rms = lb.feature.rms(y=y, sr=sr)[0]
    rms_stats = get_stats(rms, time_domain_features)
    return np.concatenate((zcr_stats, rms_stats))

def get_frequency_domain_features(y, sr, duration=6.0):
    # Frequency domain
    # magnitude STFT
    stft = np.abs(lb.stft(y, n_fft=sr, hop_length=sr//2))

    # Features
    frequency_domain_features = np.zeros(0)
    spec_centroid = lb.feature.spectral_centroid(S=stft, sr=sr)[0]
    centroid_stats = get_stats(spec_centroid, frequency_domain_features)

    spec_bw = lb.feature.spectral_bandwidth(S=stft, sr=sr)[0]
    bw_stats = get_stats(spec_bw, frequency_domain_features)
    spec_rolloff = lb.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.85)[0]
    rolloff_stats = get_stats(spec_rolloff, frequency_domain_features)

    mfcc = lb.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    return np.concatenate((centroid_stats, bw_stats, rolloff_stats, mfcc_mean, mfcc_std))

def build_dataset(folder, sr=22050, duration=6.0):
    get_time_domain_features()


def read_files(folder, vehicle):
    for dir in os.listdir(f"normalized/{vehicle}/"):
        if dir == '.DS_Store':
            continue
        for file in os.listdir(f"normalized/{vehicle}/{dir}"):
            if file == '.DS_Store':
                continue