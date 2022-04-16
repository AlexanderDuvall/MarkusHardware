import pyaudio
import wave
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import time
import board
# import busio
# import adafruit_bme280
# import adafruit_gps
# import serial
import threading
import os.path
import tensorflow as tf
from tensorflow import keras
from pydub import AudioSegment
import scipy
import scipy.signal
import scipy.misc
from os import listdir
import time as t
import soundfile as sf
import queue
import cv2
import tensorflow_io as tfio

from threading import Thread
import queue

isExperimental = False

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=2000)])
logical_gpus = tf.config.list_logical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
counter = 0
avg = 0
# GLOBAL RUNNING VARIABLE FOR THREAD MANAGEMENT
done = False

# GLOBAL VARIABLE FOR IMAGES TO SAVE TIME
axes = plt.gca()


# Time to string formatting
def gettime(aTime):
    m, s = divmod(aTime, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    w, d = divmod(d, 7)
    return f'{w:02.0f}:{d:02.0f}:{h:02.0f}:{m:02.0f}:{s:07.4f}'.format(w=w, d=d, h=h, m=m, s=s)


# Progress bar
def progressBar(cap, startTime, w):
    global count
    prevCount = count
    progress = (prevCount * 100) / cap
    filled = prevCount * w // cap
    bars = "#" * int(filled)
    gap = " " * (w - filled)
    dis = f'|{bars}{gap}| ({count}/{cap}) {progress:05.2f}% ' \
        .format(bars=bars, gap=gap, progress=progress, count=prevCount, cap=cap)

    time = gettime(t.time() - startTime)
    cur = f'Time Elapsed: {time}'.format(time=time)
    print(dis + cur, end='')

    while prevCount < cap:
        if prevCount != count:
            prevCount = count

        if prevCount % (cap // w) == 0 and prevCount > 0:
            filled = prevCount * w // cap
            bars = "#" * filled
            gap = " " * (w - filled)

        progress = (prevCount * 100) / cap
        backspace = '\b' * (len(dis) + len(cur))
        dis = f'|{bars}{gap}| ({count}/{cap}) {progress:05.2f}% ' \
            .format(bars=bars, gap=gap, progress=progress, count=prevCount, cap=cap)
        time = gettime(t.time() - startTime)
        cur = f'Time Elapsed: {time}'.format(time=time)
        print(backspace + dis + cur, end='')

    print()


# Thread function to save files
def fileWriter(q):
    global count
    while not done or not q.empty():
        if not q.empty():
            # Save waiting images
            data = q.get()
            cv2.imwrite(data[0], data[1])

            # Reset the plot to save memory every 250 items
            if count % 250 == 0:
                axes.cla()
            count += 1


# Modified/Simplified version of librosa.load
def readFile(filename):
    # Load the file
    context = sf.SoundFile(filename + ".wav")
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
# dirIn Read File
# dirOut where to output file
# filename name of file
# q Queue for threading
def experiMelSpec(dirIn, dirOut, filename, q):
    # Check if the image exists already
    test = dirOut + filename + ".png"
    my_file = Path(test)
    if not my_file.is_file():
        # Read the file
        start = t.time()
        y, sr = readFile(dirIn + filename)
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
        # q.put([test, image])
        plt.close()
        return load, calc, t.time() - start
    else:
        # Remove the image and run the algorithm
        os.remove(test)
        return experiMelSpec(dirIn, dirOut, filename, q)


# will load the model and return it. One time setup only
# The directory must point to the variables AND the model file like so
# Saved Model Data\\MarkusModel\\
def audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 1.5

    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()
    if os.path.isfile(WAVE_OUTPUT_FILENAME) == False:
        open(WAVE_OUTPUT_FILENAME, 'x')
        print("Creating file....\n")
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def load(modelPath):
    global model
    # filePath = "/home/allen/Desktop/pyPro/Master/Saved_Model_Data//MarkusModel"
    model = tf.keras.models.load_model(modelPath)
    return model


##
# Will predict the outcome of a target image in the directory.
# directory must be like: C:\\Markus\\melSpecFiles\\ZOOM0011204.png
##
def predictOutcome(model, imageDirectory):
    # filePath = "Saved Model Data\\MarkusModel\\"
    # model = keras.models.load_model(filePath)
    # model = load("Saved Model Data\\MarkusModel\\")
    img = keras.preprocessing.image.load_img(
        imageDirectory, target_size=(314, 235)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    print("getting..")
    predictions = model(img_array, training=False)
    # predictions = model.predict(img_array, batch_size=None ,verbose=0,steps=None, callbacks=None,max_queue_size=10,workers=64,use_multiprocessing=True)
    print("done...")
    score = tf.nn.softmax(predictions[0])
    print(score)
    return score


# model = load("/home/allen/Desktop/pyPro/Master/Saved_Model_Data/MarkusModel")


def makeMelSpec(ai, filename):
    test = filename + ".png"
    my_file = Path(test)

    if my_file.is_file():
        os.remove(test)
    y, sr = librosa.load(filename + ".wav")
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

    # variable = predictOutcome(ai, test)


def tensorData(filename):
    audio = tf.io.read_file(filename + ".wav")
    # audio = tfio.audio.AudioIOTensor("C:\\Markus\\Users\\" + str(id) + "\\" + filename + ".wav")
    audio2 = tf.audio.decode_wav(audio, desired_channels=1)
    audio_slice = audio2[0]  # get tensor instead of array
    # audio_tensor = tf.squeeze(audio_slice,[0])
    audio_slice2 = tf.cast(audio_slice, tf.float32)
    spectrogram = tfio.audio.spectrogram(
        audio_slice, nfft=512, window=512, stride=256)
    mel_spectrogram = tfio.audio.melscale(
        spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)
    dbscale_mel_spectrogram = tfio.audio.dbscale(mel_spectrogram, top_db=80)
    return dbscale_mel_spectrogram


# audio data location goes here

# makeMelSpec("/home/allen/Desktop/pyPro/output")


# while true
# record audio data 1.5 sec
# save recording
# make melspec (audio file location, png file)
# v /outputdata/ png file location

# print(variable)

def setup():
    model = load("Markus-master/Saved Model Data/MarkusModel")
    return model


def startAi(ai):
    global avg
    audio()
    start = time.time()
    q = queue.Queue()
    if isExperimental:
        experiMelSpec("C:\\Users\\alexa\\PycharmProjects\\MarkusHardware\\", "\\", "output", q)
        # makeMelSpec(ai, "output")
    else:
        tensorData("output")
    stop = time.time()
    avg += (start - stop)


def model():
    model = setup()
    global counter
    global avg
    global isExperimental
    for x in range(2):
        while counter != 100:
            # start thread
            startAi(model)
            counter += 1
        print("Average Time " + str(avg / counter))
        isExperimental = True
        counter = 0
        avg = 0


# def temp():
#    i2c = busio.I2C(board.SCL, board.SDA)
#    bme280 = adafruit_bme280.Adafruit_BME280_I2C(i2c)
#
#    bme280.sea_level_pressure = 1013.25
#    bme280.mode = adafruit_bme280.MODE_NORMAL
#    bme280.standby_period = adafruit_bme280.STANDBY_TC_500
#    bme280.iir_filter = adafruit_bme280.IIR_FILTER_X16
#    bme280.overscan_pressure = adafruit_bme280.OVERSCAN_X16
#    bme280.overscan_humidity = adafruit_bme280.OVERSCAN_X1
#    bme280.overscan_temperature = adafruit_bme280.OVERSCAN_X2
#
#    time.sleep(1)
#
#    while True:
#        print("\nTemperature: %0.1f C" % bme280.temperature)
#        print("Humidity: %0.1f %%" % bme280.relative_humidity)
#        print("Altitude = %0.2f meters" % bme280.altitude)
#        print("Pressure: %0.1f hPa" % bme280.pressure)
#        time.sleep(1)


# def gps():
#    uart = serial.Serial("/dev/ttyTHS1", baudrate=9600, timeout=10)
#
#    gps = adafruit_gps.GPS(uart, debug=False)
#
#    gps.send_command(b"PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")
#
#    gps.send_command(b"PMTK220,1000")
#
#    last_print = time.monotonic()
#    count = 1
#
#    while True:
#        gps.update()
#
#        current = time.monotonic()
#        if current - last_print >= 1.0:
#            last_print = current
#            if not gps.has_fix:
#                print("Waiting for fix...")
#                continue
#
#            print("=" * 40)  # Print a separator line.
#            print(
#                "Fix timestamp: {}/{}/{} {:02}:{:02}:{:02}".format(
#                    gps.timestamp_utc.tm_mon,  # Grab parts of the time from the
#                    gps.timestamp_utc.tm_mday,  # struct_time object that holds
#                    gps.timestamp_utc.tm_year,  # the fix time.  Note you might
#                    gps.timestamp_utc.tm_hour,  # not get all data like year, day,
#                    gps.timestamp_utc.tm_min,  # month!
#                    gps.timestamp_utc.tm_sec,
#                )
#            )
#            print("Latitude: {0:.6f} degrees".format(gps.latitude))
#            print("Longitude: {0:.6f} degrees".format(gps.longitude))
#            print("Fix quality: {}".format(gps.fix_quality))
#
#            if gps.satellites is not None:
#                print("# satellites: {}".format(gps.satellites))
#            if gps.altitude_m is not None:
#                print("Altitude: {} meters".format(gps.altitude_m))
#            if gps.speed_knots is not None:
#                print("Speed: {} knots".format(gps.speed_knots))
#            if gps.track_angle_deg is not None:
#                print("Track angle: {} degrees".format(gps.track_angle_deg))
#            if gps.horizontal_dilution is not None:
#                print("Horizontal dilution: {}".format(gps.horizontal_dilution))
#            if gps.height_geoid is not None:
#                print("Height geo ID: {} meters".format(gps.height_geoid))
#

# global data for sensors ie temp= t gps = g, ...
# t4 for jetsondata for sending off
# every x secs... 10 secs


# t1 = threading.Thread(target=model, args=())
# t2 = threading.Thread(target=temp, args=())
# t3 = threading.Thread(target=gps, args=())

# t1.start()
# t2.start()
# t3.start()

##t2 = time.sleep(10)
##t3 = time.sleep(10)
##
# t1.join()
##t2.join()
##t3.join()
model()
