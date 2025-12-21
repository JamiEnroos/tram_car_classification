import os
import librosa as lb
import numpy as np
import soundfile as sf
from matplotlib.ticker import MaxNLocator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

"""
Binary sound source classifier for cars and trams using support vector machine.

The project directory structure should be separated into test, train and validation data. Data should be in the data
directory. The normalize function writes normalized files into the normalized/ directory as .wav files.
The normalized/ and its subdirs are created automatically if they don't exist.

main.py
data/
        test/
                car/
                        audio_pack/ <--- corresponding audio should be in dirs under the vehicle
                        audio_file.wav  <-- or they can be here as well
                        ....
                tram/
                        audio_pack/
                        ...
        train/
                car/
                        audio_pack/     and same for all vehicle subdirs in data
                        ...
                tram/
        val/
                car/
                tram/


normalized/         <--- created automatically when normalizing
        test/
                car/
                tram/
        train/
                car/
                tram/
        val/
                car/
                tram/
                

In results tram is indicated with 0 and car with 1.
"""

def normalize(vehicle, base_dir="data", wr_dir="normalized"):
    """
    This function will normalize audio signals and write them as .wav files into wr_dir/...
    With this function we need to normalize the audio signals only once. After that we can run the model as
    many times as we want.

    vehicle : string, name of the vehicle subdirectory that we want to normalize (under data/train;test;val)
    base_dir : string, directory name where to read data from
    wr_dir : string, directory name to write normalized audio files into
    """
    for subdir in os.listdir(base_dir): # test/, train/ and val/
        subpath = os.path.join(base_dir, subdir)
        if not os.path.isdir(subpath):
            continue

        vehicle_dir = os.path.join(subpath, vehicle) # test/{vehicle} ...
        if not os.path.isdir(vehicle_dir):
            continue

        for user_dir in os.listdir(vehicle_dir): # sound pack dir, test/{vehicle}/{pack}
            subpath = os.path.join(vehicle_dir, user_dir)
            if not os.path.isdir(subpath):
                subpath = vehicle_dir

            for filename in os.listdir(subpath):
                if not filename.lower().endswith((".wav", ".mp3", ".m4a")):
                    continue

                full_path = os.path.join(subpath, filename)
                try:
                    audio, fs = lb.load(full_path)

                    audio = lb.util.normalize(audio)

                    wr_path = f"{wr_dir}/{subdir}/{vehicle}/{filename[0:-4]}_norm.wav"
                    os.makedirs(os.path.dirname(wr_path), exist_ok=True) # Create the output dirs if they don't exist
                    sf.write(wr_path, audio, fs)

                except Exception as e:
                    print(f"Failed to load {filename}: {e}")


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


def get_stats(x):
    """
    Returns mean, std of given array x as one numpy array
    """
    mean = np.mean(x)
    std = np.std(x)
    return np.array([mean, std])

def get_time_domain_features(y):
    """
    Returns ZCR mean & std and RMS mean & std as one array.
    """
    time_domain_features = np.zeros(0)
    # Time domain
    zcr = lb.feature.zero_crossing_rate(y)[0]
    zcr_stats = get_stats(zcr)

    rms = lb.feature.rms(y=y)[0]
    rms_stats = get_stats(rms)
    return np.concatenate((zcr_stats, rms_stats))

def get_frequency_domain_features(y, sr, plot_spectrogram):
    """
    Returns frequency domain features of given signal y as one numpy array
    y: array, input signal
    sr: int, sampling rate
    plot_spectrogram : str, "None" - no plots, "Car" - plots MFCC and spectrogram of car,
                            "Tram" - plots MFCC and spectrogram of tram
    """
    # Frequency domain
    hop_length = 512
    n_fft = 2048
    # magnitude STFT
    stft = np.abs(lb.stft(y, n_fft=n_fft, hop_length=hop_length))

    # Features
    spec_centroid = lb.feature.spectral_centroid(S=stft, sr=sr)[0]
    centroid_stats = get_stats(spec_centroid)

    spec_bw = lb.feature.spectral_bandwidth(S=stft, sr=sr)[0]
    bw_stats = get_stats(spec_bw)

    spec_rolloff = lb.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.85)[0]
    rolloff_stats = get_stats(spec_rolloff)

    mfcc = lb.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1) # 13 values
    mfcc_std = mfcc.std(axis=1)   # 13 values

    if plot_spectrogram != "None":
        #print("Plotting spectrogram and MFCC")
        plt.figure()
        lb.display.specshow(lb.amplitude_to_db(stft), y_axis='log', x_axis='time', sr=sr, hop_length=hop_length)
        plt.title(f"{plot_spectrogram} Spectrogram")
        plt.colorbar()
        plt.show()

        lb.display.specshow(mfcc[1:], x_axis='time', sr=sr)
        plt.ylabel("MFCC coefficient (1-12)")
        plt.yticks()
        plt.title(f"{plot_spectrogram} MFCCs")

        # Show MFCC coefficient numbers (integers 0-12)
        #plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True)) # start from 0

        plt.colorbar()
        plt.show()

    return np.concatenate((centroid_stats, bw_stats, rolloff_stats, mfcc_mean, mfcc_std))


def build_dataset(base_dir = "normalized"):
    """
    Organizes data into train, validation and test sets.
    base_dir : string, directory name where to read data from
    returns: train, validation and test sets
    """
    X_train_car, y_train_car, X_test_car, y_test_car, X_val_car, y_val_car = read_files(base_dir, "car")
    X_train_tram, y_train_tram, X_test_tram, y_test_tram, X_val_tram, y_val_tram = read_files(base_dir, "tram")

    X_train = np.vstack([X_train_car, X_train_tram])  # stack rows (files)
    y_train = np.concatenate([y_train_car, y_train_tram])  # stack labels
    X_test = np.vstack([X_test_car, X_test_tram])
    y_test = np.concatenate([y_test_car, y_test_tram])
    X_val = np.vstack([X_val_car, X_val_tram])
    y_val = np.concatenate([y_val_car, y_val_tram])

    return X_train, X_test, X_val, y_train, y_test, y_val

def support_vector_classifier(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    The actual SVM model.
    Finds the best parameters using the sklearn pipeline.
    """

    # Combine Training and Validation data to use the GridSearchCV function
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.hstack((y_train, y_val))

    # Create a list where -1 indicates that the sample is from the training set and 0 for the validation set.
    # -1s
    train_indices = np.full((len(X_train),), -1, dtype=int)

    # 0s
    val_indices = np.full((len(X_val),), 0, dtype=int)

    # Combine them
    test_fold = np.concatenate((train_indices, val_indices))

    # Create the PredefinedSplit object, since data was manually split
    ps = PredefinedSplit(test_fold)

    # Scale the features so that they affect the classification equally
    scaler = StandardScaler()

    # Params for the SVClassifier
    params = {
        'svc__C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
        'svc__gamma': ['scale'],
        'svc__kernel': ['rbf', 'linear']
    }

    pipe = make_pipeline(
        scaler,
        SVC(kernel='rbf', probability=False) # Starting kernel
    )

    # Finds the best parameters with cross validation
    grid = GridSearchCV(pipe, param_grid=params, cv=ps, verbose=1)

    grid.fit(X_combined, y_combined)

    print(f"Best Parameters found: {grid.best_params_}")
    y_pred = grid.predict(X_test)

    print("\nFinal Test Set Results:")
    print(classification_report(y_test, y_pred))

    """ 
        Test if data values are similar
        print()
        print(f"Train Data Mean: {np.mean(X_train):.4f}")
        print(f"Test Data Mean: {np.mean(X_test):.4f}")
        print(f"Validation Data Mean: {np.mean(X_val):.4f}\n")

        print(f"Train Data Std: {np.std(X_train):.4f}")
        print(f"Test Data Std: {np.std(X_test):.4f}")
        print(f"Validation Data Std: {np.std(X_val):.4f}\n")

        print("Train Max:", np.max(X_train))
        print("Test Max: ", np.max(X_test))
    """

def read_files(base_dir, vehicle, duration=6.0, sr =22050):
    """
    Read all .wav files in given directory and
    return: split data as numpy arrays.

    """

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    val_features = []
    val_labels = []

    current_label = 100

    zcr_means = []
    rms_means = []

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

            plot_spectrogram = "None"

            # Plot these files
            if filename == "656929__aliabdelsalam__car-10_norm.wav":
                plot_spectrogram = "Car"
                #print(filename, "FILE FOUND \n")
            elif filename == "tram 14_norm.wav":
                plot_spectrogram = "Tram"
                #print(filename, "FILE FOUND \n")


            y, sr = load_audio(os.path.join(full_path), sr=sr, duration=duration, mono=True)
            freq_features = get_frequency_domain_features(y, sr, plot_spectrogram=plot_spectrogram) # should be 1D
            time_features = get_time_domain_features(y)

            zcr_means.append(time_features[0])
            rms_means.append(time_features[2])

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

    print("Train Label count: ", vehicle, len(train_labels))
    print("Test Label count: ", vehicle, len(test_labels))
    print("Validation Label count: ", vehicle, len(val_labels), "\n")

    print(f"{vehicle} zcr mean mean: {np.mean(zcr_means):.4f}")
    print(f"{vehicle} rms mean mean: {np.mean(rms_means):.4f}\n")

    X_train = np.vstack(train_features)
    y_train = np.array(train_labels)
    X_test = np.vstack(test_features)
    y_test = np.array(test_labels)
    X_val = np.vstack(val_features)
    y_val = np.array(val_labels)
    return X_train, y_train, X_test, y_test, X_val, y_val

def main():
    # Normalize automatically if dir normalized/train/car doesn't exist. However, you should call normalize manually
    # if any of the needed directories don't exist
    if not os.path.isdir("normalized/train/car"):
        normalize(vehicle="car")
        normalize(vehicle="tram")

    X_train, X_test, X_val, y_train, y_test, y_val = build_dataset()
    support_vector_classifier(X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == "__main__":
    main()