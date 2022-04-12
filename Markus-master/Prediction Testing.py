import tensorflow as tf
import tensorflow.keras as keras
from flask import Flask, request, jsonify
import requests
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.allow_growth = True

# will load the model and return it. One time setup only
# The directory must point to the variables AND the model file like so
# Saved Model Data\\MarkusModel\\
def load(modelPath):
    global model
    # filePath = "Saved Model Data\\MarkusModel\\"
    model = keras.models.load_model(modelPath)
    return model


def makeMelSpec(id, filename):
    test = "C:\\Markus\\Users\\" + str(id) + "\\" + filename + ".png"  # where file is saved
    my_file = Path(test)
    if not my_file.is_file():
        y, sr = librosa.load("C:\\Markus\\Users\\" + str(id) + "\\" + filename + ".wav")  # where wave file is stored
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


model = load("Saved Model Data\\MarkusModel\\")


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
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    labeled = {'Crowd_Conversation': score[0], 'Large_Conversation': score[1], "Noise": score[2], "Quiet": score[3],
               "Small_Conversation": score[4], "Stimuli": score[5], "Unknown": score[6], "Wind": score[7]}

    return labeled


app = Flask("app")


@app.route("/test", methods=["GET"])
def d():
    info = request.get_json()
    userid = info["id"]
    makeMelSpec(userid, "soundfile")
    label = predictOutcome(model, "C:\\Markus\\Users\\" + str(userid) + "\\" + "soundfile.png")
    prediction = ""
    for key, value in label.items():
        prediction += str(key) + ":" + str(value) + "\n"
    return ({"data": prediction})


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=9696)

# model = load("Saved Model Data\\MarkusModel\\")
# label = (predictOutcome(model, "C:\\Users\\Alex\\OneDrive\\ZOOM0013156.png"))
# for key, value in label.items():
#    print(key, ' : ', value)
