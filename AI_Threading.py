# https://github.com/jiaaro/pydub
from pathlib import Path

from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import time as t
import scipy
import scipy.signal
import scipy.misc
from os import listdir
import time as t
import soundfile as sf
import time
import cv2
import math
from threading import Thread
import queue
import tensorflow as tf
from tensorflow import keras

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=165)])
logical_gpus = tf.config.list_logical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

count = 0
# GLOBAL RUNNING VARIABLE FOR THREAD MANAGEMENT
done = False
q = queue.Queue()
# GLOBAL VARIABLE FOR IMAGES TO SAVE TIME
axes = plt.gca()
count = 0
model = 0


def load(modelPath):
    global model
    # filePath = "/home/allen/Desktop/pyPro/Master/Saved_Model_Data//MarkusModel"
    model = tf.keras.models.load_model(modelPath)
    return model


def setup():
    model = load("C:\\Users\\alexa\\Markus\\Saved Model Data\\MarkusModel")
    return model


# Thread function to save files
def clearMemory(threadList):
    global count
    global done
    print("Prioritizing Memory...")
    while not done:
        deadCheck = True
        print(count)
        if count > 250:
            axes.cla()
            print("clear memory...")
            count = 0
        for thread in threadList:
            if thread.is_alive():
                deadCheck = False
        if deadCheck:
            break
        time.sleep(5)
    print("We're done here!")
    axes.cla()


def clearMemoryNoThread():
    global count
    global done
    print("Prioritizing Memory...")
    print(count)
    if count > 12:
        axes.cla()
        print("clear memory...")
        count = 0
    time.sleep(5)


print("We're done here!")
axes.cla()


# Modified/Simplified version of librosa.load
def readFile(filename):
    # Load the file
    context = sf.SoundFile(filename)
    y = context.read(frames=-1, dtype=np.float32, always_2d=False).T

    # Shift the structure to match the ML Algorithm
    y = np.mean(y, axis=tuple(range(y.ndim - 1)))
    sr_native = context.samplerate
    samples = round(len(y) * float(22050) / sr_native)
    f = scipy.signal.resample(y, samples, axis=-1)

    return f, 22050


# Simplify librosa.display.specshow
def generateImage(DB, sr, hop_length, **kwargs):
    # Set up proper arguments
    kwargs.setdefault("cmap", librosa.display.cmap(DB))
    kwargs.setdefault("rasterized", True)
    kwargs.setdefault("edgecolors", "None")
    kwargs.setdefault("shading", "auto")

    # Generate the image (This is the slowest part now)
    y = np.arange(DB.shape[0])
    x = np.arange(DB.shape[1])
    out = axes.pcolormesh(x, y, DB, **kwargs)
    return out


# New algorithm
# NOTE : Anything related to t.time() isn't necessary to the code
#        but if you want to test how long it takes, keep the timers
#        there. When producing the final device, remove them, as
#        they take (minimal) time to actually happen. You can make
#        the function be a void, return values are unnecessary.
def experiMelSpec(dirIn, dirOut, filename):
    global count
    # Check if the image exists already
    test = dirOut + filename + ".png"
    my_file = Path(test)
    # Read the file
    start = t.time()
    y, sr = readFile(dirIn + os.sep + filename)
    load = t.time() - start
    # trim silent edges
    start = t.time()
    whale_song, _ = librosa.effects.trim(y)
    # Calculate the short-time fourier transform
    n_fft = 2048
    hop_length = 100
    D = np.abs(librosa.stft(whale_song, n_fft=n_fft, hop_length=hop_length + 1))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    calc = t.time() - start
    start = t.time()
    # Export the image using OpenCV
    qm1 = generateImage(DB, sr, hop_length + 1)
    arr2 = 255 * qm1.to_rgba(qm1.get_array().reshape(np.shape(DB)))
    arr2[:, :, [0, 2]] = arr2[:, :, [2, 0]]
    image = cv2.resize(arr2, (640, 480))
    image = cv2.flip(image, 0)
    # Add item to the other thread
    cv2.imwrite(dirOut + os.sep + "predict.png", image)
    plt.close()
    count += 1
    return load, calc, t.time() - start


def predictDLOutcome(fileLocation, OutputDirectory, Filename, DeepLearning_Model):
    experiMelSpec(fileLocation, OutputDirectory, Filename)


def prepareAndLaunchThreads(list, audioDirectory, outputDirectory):
    listOfThreads = []
    memoryThread = Thread(target=clearMemory, args=(listOfThreads,))
    memoryThread.start()

    for thread in listOfThreads:
        thread.join()
    memoryThread.join()
    return listOfThreads


def bulkExportMelSpec(outputDirectory, audioFilesDirectory):
    # Start the file writing thread here

    global done
    fileList = []
    fileList.extend(listdir(audioFilesDirectory))
    splitFiles = np.array_split(fileList, 4)

    threadList = prepareAndLaunchThreads(splitFiles, audioFilesDirectory, outputDirectory)
    # for x in os.listdir(audioFilesDirectory):
    #    _, _, s = experiMelSpec(audioFilesDirectory, outputDirectory, x)
    #    print(s)
    done = True
    # thread.join()
    print('Done.')


def predictOutcome(model, imageDirectory):
    # filePath = "Saved Model Data\\MarkusModel\\"
    # model = keras.models.load_model(filePath)
    # model = load("Saved Model Data\\MarkusModel\\")
    img = keras.preprocessing.image.load_img(
        imageDirectory, target_size=(314, 235)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model(img_array, training=False)
    # predictions = model.predict(img_array, batch_size=None ,verbose=0,steps=None, callbacks=None,max_queue_size=10,workers=64,use_multiprocessing=True)
    # score = tf.nn.softmax(predictions[0])
    # labeled = {'Crowd_Conversation': score[0], 'Large_Conversation': score[1], "Noise": score[2], "Quiet": score[3],
    #           "Small_Conversation": score[4], "Stimuli": score[5], "Unknown": score[6], "Wind": score[7]}
    # prediction = ""
    # for key, value in labeled.items():
    #    prediction += str(key) + ":" + str(value) + "\n"
    # print(prediction)
    # return labeled


def launchSpectrograms(listofFiles, audioDirectory, outputDirectory):
    global count
    timeTable = ""
    print("show time!!")
    for file in listofFiles:
        start = t.time()
        experiMelSpec(audioDirectory, outputDirectory, file)
        melspecTime = t.time()
        start2 = t.time()
        predictOutcome(model, "C:\\Users\\alexa\\Documents\\JetsonFiles\\" + os.sep + "predict.png")
        predictionTime = t.time()
        if (count > 50):
            print("clearing memory")
            clearMemoryNoThread()
        timeTable += str(start - melspecTime) + "," + str(start2 - predictionTime) + "\n"
    print(timeTable)
    # exit(1)


if __name__ == '__main__':
    # cutAll()
    type = '.wav'
    # for root, dirs, files in os.walk("C:\\Users\\alexa\\Documents\\USB Shit\\Markus Audio"):
    #     for name in files:
    #         if (name.lower().endswith(type) and not "melspecfiles" in root.lower()
    #                 and not "audiofiles" in root.lower()):
    #             cutAll(root, "\\" + name, type)
    # start = time.time()
    # bulkExportMelSpec("D:\\Markus 2\\Markus Audio", "C:\\Markus\\audioFiles")
    # end = time.time()
    # print(start - end)
    model = setup()
    for root, dirs, files in os.walk("C:\\Markus Project\\audioFiles"):
        launchSpectrograms(files, root, "C:\\Users\\alexa\\Documents\\JetsonFiles")
