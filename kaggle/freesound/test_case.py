import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import os

DIR = "../../../input/audio_test/"
SAMPLE_RATE = 44100
FIX_SAMPLE = 256


def print_single_file_log_mel(dir, file):
    # Load sound file
    y, sr = librosa.load(os.path.join(dir, file), sr=SAMPLE_RATE)

    duration = y.shape[0] / sr
    print(file + " - duration=", duration)
    offset = 0
    if duration > 20:
        offset = 10
    elif duration > 14:
        offset = 7
    elif duration > 8:
        offset = 4
    elif duration > 4:
        offset = 2
    if offset > 0:
        y, sr = librosa.load(os.path.join(dir, file), sr=SAMPLE_RATE, offset=offset)
        duration = y.shape[0] / sr
        print(file + " - duration=", duration)

    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.amplitude_to_db(S, ref=np.max)

    print(file + " - log_S.shape=", log_S.shape)

    sec_dim = log_S.shape[1]
    if sec_dim > FIX_SAMPLE:
        log_S = log_S[:, :FIX_SAMPLE]
    elif sec_dim < FIX_SAMPLE:
        log_S = np.pad(log_S, ((0, 0), (0, FIX_SAMPLE - sec_dim)), 'constant', constant_values=(0, 0))

        # Make a new figure
    plt.figure(figsize=(10, 4))

    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

    # Put a descriptive title on the plot
    plt.title(file + ' - mel power spec')

    # draw a color bar
    plt.colorbar(format='%+02.0f dB')

    # Make the figure layout compact
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    for root, dirs, files in os.walk(DIR):
        for i in range(110, 120):
            print_single_file_log_mel(root, files[i])
