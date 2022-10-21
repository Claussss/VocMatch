# -----------------------------------------------------------
# Generates normalized spectograms from the audio files located in the specified directory.
# The spectograms are saved as .png images in the result directory.
# -----------------------------------------------------------



from ctypes.wintypes import HLOCAL
import os
import librosa
import librosa.display
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

result_path = r'data\gram_dataset'
path_to_dataset = r'data\audio_dataset'
DEFAULT_SAMPLE_RATE = 22050
FRAME_SIZE = 512
HOP_LENGTH = 256
MONO = True


def save_plotted_spectogram(spectogram, save_path):
    fig = plt.figure(figsize=(0.72, 0.72))
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    librosa.display.specshow(spectogram) # , sr=DEFAULT_SAMPLE_RATE, hop_length=HOP_LENGTH

    plt.savefig(save_path, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close('all')


def extract_spectogram(signal):
    """Extract log spectogram (chromagram) from the signal.
       Args:
        - signal: the input signal
        - chorma: if true, convert the spectogram to a chromagram representation
       Returns:
          Spectogram (if chroma is true, chromagram is returned)
    """
    stft = librosa.stft(signal,
                        n_fft=FRAME_SIZE,
                        hop_length=HOP_LENGTH)[:-1]
    spectogram = np.abs(stft)
    return spectogram


def extract_chromagram(spectogram):
    """Extract chromogram from the specrogram."""
    chromagram = librosa.feature.chroma_stft(S=spectogram, sr=DEFAULT_SAMPLE_RATE)
    return chromagram


def extract_tempogram(signal):
    """Extract tempogram from the signal."""
    oenv = librosa.onset.onset_strength(y=signal, sr=DEFAULT_SAMPLE_RATE, hop_length=HOP_LENGTH)
    tempogram = librosa.feature.tempogram(
        onset_envelope=oenv, sr=DEFAULT_SAMPLE_RATE, hop_length=HOP_LENGTH)
    return tempogram


def min_max_normalization(arr):
    """Normalizes an array between 0 and 1."""
    arr_min = arr.min()
    return (arr - arr_min) / (arr.max() - arr_min)


if __name__ == '__main__':
    dir_names = os.listdir(path_to_dataset)
    # Iterate though train, valid, test datasets
    for dir_name in dir_names:
        print(f'\n ==={dir_name}=== \n')
        full_path_to_result_dir = os.path.join(result_path,dir_name)
        if not os.path.exists(full_path_to_result_dir):
            os.mkdir(full_path_to_result_dir)

        full_path_to_curr_dir = os.path.join(path_to_dataset,dir_name)
        file_names = os.listdir(full_path_to_curr_dir)
        # Iterate through files
        for file_name in tqdm(file_names):
            full_path_to_file = os.path.join(full_path_to_curr_dir, file_name)
            # Load the audio signal in mono
            audio, _ = librosa.load(full_path_to_file, sr=DEFAULT_SAMPLE_RATE, mono=MONO)
            spectogram = extract_spectogram(audio)
            # Visual representations of the audio signal. 
            # Namely, it contains spectogram, chromagram, and tempogram
            visual_representations = {}
            visual_representations['spectogram'] = librosa.amplitude_to_db(spectogram) # convert it to db 
            visual_representations['chromagram'] = librosa.amplitude_to_db(extract_chromagram(spectogram))
            visual_representations['tempogram'] = extract_tempogram(audio)

            for repr_name, v_repr in visual_representations.items():
                full_path_to_repr = os.path.join(full_path_to_result_dir, repr_name)
                if not os.path.exists(full_path_to_repr):
                    os.mkdir(full_path_to_repr)
                full_path_to_result_file = os.path.join(full_path_to_repr, file_name.replace('.wav', '.png'))

                save_plotted_spectogram(min_max_normalization(v_repr), full_path_to_result_file)

