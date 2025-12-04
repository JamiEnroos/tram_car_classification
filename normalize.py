import os

import librosa as lb
import soundfile as sf


def normalize(vehicle):
    i = 1
    for dir in os.listdir(f"data/{vehicle}/"):
        if dir == '.DS_Store':
            continue
        for file in os.listdir(f"data/{vehicle}/{dir}"):
            if file == '.DS_Store':
                continue
            audio, fs = sf.read(f"data/{vehicle}/{dir}/{file}")
            print(f"before normalization: {audio}")
            audio = lb.util.normalize(audio)
            print(f"after: {audio}")
            sf.write(f"normalized/{vehicle}/{vehicle}_{i}.wav", audio, fs)
            # sf.write(f"normalized/{vehicle}/file.wav", audio, fs)

            i += 1

normalize("car")
normalize("tram")
