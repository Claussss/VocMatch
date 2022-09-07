# -----------------------------------------------------------
# Randomly applies the following audio signal augmentations on the all the files in the directory:
#  - pitch scale
#  - time stretch
#  - adding white noise
# and saves newly augmented audio files in the same directory.
# -----------------------------------------------------------


import os
import random

import librosa
from tqdm import tqdm
import soundfile as sf
import numpy as np

path_to_files = r'D:\Education\Projects\FiZam\data\audio_dataset\train'
path_to_results = r'D:\Education\Projects\FiZam\data\audio_dataset\train'

WHITE_NOISE_CHANCE = 0.6
TIME_STRETCH_CHANCE = 0.8
PITCH_SCALE_CHANCE = 0.8

NOISE_FACTOR_RANGE = (0, 0.05)
STRETCH_RATE_RANGE = (0.9, 1)
NUM_SEMITONS_RANGE = (-3, 3)


def add_white_noise(signal, noise_factor):
    noise = np.random.normal(0, signal.std(), len(signal))
    augmented_signal = signal + noise * noise_factor
    return augmented_signal


def time_stretch(signal, stretch_rate):
    return librosa.effects.time_stretch(signal, stretch_rate)


def pitch_scale(signal, sr, num_semitones):
    return librosa.effects.pitch_shift(signal, sr, num_semitones)


def random_white_noise(signal, sr):
    if random.random() < WHITE_NOISE_CHANCE:
        noise_factor = np.random.uniform(*NOISE_FACTOR_RANGE)
        signal = add_white_noise(signal, noise_factor)
        new_file_name = file_name[:-4] + f'_white_noise_{noise_factor:.2f}' + '.wav'
        sf.write(os.path.join(path_to_results, new_file_name), signal, sr)
    return signal


def random_time_stretch(signal, sr):
    if random.random() < TIME_STRETCH_CHANCE:
        stretch_rate = np.random.uniform(*STRETCH_RATE_RANGE)
        signal = time_stretch(signal, stretch_rate)
        new_file_name = file_name[:-4] + f'_time_stretch_{stretch_rate:.2f}' + '.wav'
        sf.write(os.path.join(path_to_results, new_file_name), signal, sr)
    return signal


def random_pitch_scale(signal, sr):
    if random.random() < PITCH_SCALE_CHANCE:
        num_semitones = random.randint(*NUM_SEMITONS_RANGE)
        signal = pitch_scale(signal, sr, num_semitones)
        new_file_name= file_name[:-4] + f'_pitch_scale_{num_semitones}' + '.wav'
        sf.write(os.path.join(path_to_results, new_file_name), signal, sr)
    return signal


if __name__ == '__main__':
    file_names = os.listdir(path_to_files)

    for file_name in tqdm(file_names):
        full_path_to_file = os.path.join(path_to_files, file_name)
        signal, sr = librosa.load(full_path_to_file)

        # Apply augmentations in a random order
        augmentations = [random_white_noise, random_time_stretch, random_pitch_scale]
        random.shuffle(augmentations)

        for augmentation in augmentations:
            signal = augmentation(signal, sr)
