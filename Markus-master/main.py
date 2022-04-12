# https://github.com/jiaaro/pydub
from pathlib import Path

from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os


def cutSong(song, st, en, count, file_name, type, path):
    extract = song[st:en]
    # Saving
    print(path + file_name + str(count) + "WAV......")
    extract.export(path + file_name + str(count) + type, format="wav")
    return extract


def makeMelSpec(filename):
    test = "C:\\Markus\\melSpecFiles" + filename + ".png" # where file is saved
    my_file = Path(test)
    if not my_file.is_file():
        y, sr = librosa.load("C:\\Markus\\audioFiles" + filename + ".wav") # where wave file is stored
        print("C:\\Markus\\audioFiles" + filename + ".wav______________")
        # trim silent edges
        whale_song, _ = librosa.effects.trim(y)
        n_fft = 2048
        hop_length = 100
        D = np.abs(librosa.stft(whale_song, n_fft=n_fft, hop_length=hop_length + 1))
        DB = librosa.amplitude_to_db(D, ref=np.max)
        librosa.display.specshow(DB, sr=sr, hop_length=hop_length)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.show()
        plt.savefig(test, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(filename)
    else:
        print("exists")


def cutAll(files_path, file_name, type):
    try:
        os.mkdir(files_path + 'audioFiles\\')
        os.mkdir(files_path + 'melSpecFiles\\')
    except OSError as error:
        print()

    song = AudioSegment.from_wav(files_path + file_name)
    secs = song.duration_seconds
    samples = secs / 1.5
    samples = round(samples)
    if (samples >= 1):
        for i in range(samples - 1):
            d = i * 1.5
            startMin = 0
            startSec = d
            endMin = 0
            endSec = d + 1.5
            # Time to miliseconds
            startTime = startMin * 60 * 1000 + startSec * 1000
            endTime = endMin * 60 * 1000 + endSec * 1000
            name = file_name.replace(type, "")
            try:
                # cutSong(song, startTime, endTime, i, name.replace(".WAV", ""), type, 'C:\\Markus\\audioFiles\\')
                makeMelSpec(name.replace(".WAV", "") + str(i))
            except Exception as error:
                print(error)


if __name__ == '__main__':
    # cutAll()
    type = '.wav'
    for root, dirs, files in os.walk("C:\\Users\\alexa\\Documents\\USB Shit\\Markus Audio"):
        for name in files:
            if (name.lower().endswith(type) and not "melspecfiles" in root.lower()
                    and not "audiofiles" in root.lower()):
                cutAll(root, "\\" + name, type)
