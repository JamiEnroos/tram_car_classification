import os

import librosa as lb
import soundfile as sf


def normalize(vehicle, base_dir="data", wr_dir="normalized"):
    i = 1
    for subdir in os.listdir(base_dir):
        subpath = os.path.join(base_dir, subdir)
        if not os.path.isdir(subpath):
            continue

        vehicle_dir = os.path.join(subpath, vehicle)
        if not os.path.isdir(vehicle_dir):
            continue

        for user_dir in os.listdir(vehicle_dir):
            subpath = os.path.join(vehicle_dir, user_dir)
            if not os.path.isdir(subpath):
                continue
            for filename in os.listdir(subpath):
                if not (filename.lower().endswith(".wav") or filename.lower().endswith(".mp3")):
                    continue

                full_path = os.path.join(subpath, filename)

                audio, fs = sf.read(full_path)
                print(f"before normalization: {audio}")
                audio = lb.util.normalize(audio)
                print(f"after: {audio}")
                sf.write(f"{wr_dir}/{subdir}/{vehicle}/{vehicle}_{i}.wav", audio, fs)

                i += 1

if __name__ == "__main__":
    normalize("car")
    normalize("tram")
