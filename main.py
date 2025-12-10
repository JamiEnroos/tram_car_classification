# Main
import os
import librosa as lb
import numpy as np
import soundfile as sf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


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

def get_time_domain_features(y, sr, duration=6.0):
    time_domain_features = np.zeros(0)
    # Time domain
    zcr = lb.feature.zero_crossing_rate(y)[0]
    zcr_stats = get_stats(zcr, time_domain_features)

    rms = lb.feature.rms(y=y)[0]
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

def build_dataset(base_dir = "normalized"):
    X_train_car, y_train_car, X_test_car, y_test_car, X_val_car, y_val_car = read_files(base_dir, "car")
    X_train_tram, y_train_tram, X_test_tram, y_test_tram, X_val_tram, y_val_tram = read_files(base_dir, "tram")

    X_train = np.vstack([X_train_car, X_train_tram])  # stack rows (files)
    y_train = np.concatenate([y_train_car, y_train_tram])  # stack labels
    X_test = np.vstack([X_test_car, X_test_tram])
    y_test = np.concatenate([y_test_car, y_test_tram])
    X_val = np.vstack([X_val_car, X_val_tram])
    y_val = np.concatenate([y_val_car, y_val_tram])

    #print("X shape:", X.shape)  # Should be (n_files, n_features)
    #print("y shape:", y.shape)  # Should be (n_files)

    # We need to scale the feature data for SVM
    return X_train, X_test, X_val, y_train, y_test, y_val

def support_vector_classifier(X_train, y_train, X_test, y_test):

    # Scale the features so that they affect the classification equally
    scaler = StandardScaler()
    # Params for the SVC
    params = {
        'svc__C': [0.01, 0.1, 1, 10, 100],
        'svc__gamma': ['scale', 0.001, 0.01, 0.1, 1]
    }

    pipe = make_pipeline(
        scaler,
        SVC(kernel='rbf', probability=False)
    )

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=params,
        cv=5,
        n_jobs=-1,
        scoring='accuracy'
    )

    grid.fit(X_train, y_train)

    print(f"Best Parameters found: {grid.best_params_}")
    y_pred = grid.predict(X_test)

    print("\nFinal Test Set Results:")
    print(classification_report(y_test, y_pred))

def read_files(base_dir, vehicle, duration=6.0, sr =22050):

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    val_features = []
    val_labels = []

    current_label = 100


    for subdir in os.listdir(base_dir):

        subpath = os.path.join(base_dir, subdir)
        if not os.path.isdir(subpath):
            continue
        vehicle_dir = os.path.join(subpath, vehicle)
        if not os.path.isdir(vehicle_dir):
            continue
        for filename in os.listdir(vehicle_dir):

            if not filename.lower().endswith(".wav"):
                continue

            full_path = os.path.join(vehicle_dir, filename)

            y, sr = load_audio(os.path.join(full_path), sr=sr, duration=duration, mono=True)
            freq_features = get_frequency_domain_features(y, sr, duration=duration) # should be 1D
            time_features = get_time_domain_features(y, sr, duration=duration)

            combined_vec = np.concatenate([freq_features, time_features], axis=0)

            if vehicle == 'car':
                current_label = 1
            elif vehicle == 'tram':
                current_label = 0

            if current_label == 100:
                print("invalid label")
                continue

            if subdir == "train":
                train_features.append(combined_vec)

                train_labels = np.append(train_labels, current_label)
            elif subdir == "test":
                test_features.append(combined_vec)
                test_labels = np.append(test_labels, current_label)
            elif subdir == "val":
                val_features.append(combined_vec)
                val_labels = np.append(val_labels, current_label)

    X_train = np.vstack(train_features)
    y_train = np.array(train_labels)
    X_test = np.vstack(test_features)
    y_test = np.array(test_labels)
    X_val = np.vstack(val_features)
    y_val = np.array(val_labels)
    return X_train, y_train, X_test, y_test, X_val, y_val

def main():
    X_train, X_test, X_val, y_train, y_test, y_val = build_dataset()
    support_vector_classifier(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()