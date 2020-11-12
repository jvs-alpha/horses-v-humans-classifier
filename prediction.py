from PIL import Image
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def loadimage(file):
    with open(file, "rb") as f:
        img = Image.open(f.read())
    return img


if __name__ == "__main__":
    og = loadimage("human01-14.png")
    test = np.asarray(og.convert("L"))
    class_names = ["horse", "human"]
    model = keras.models.load_model("hvm_f1.h5")
    prediction = model.evaluate(test)
    plt.imshow(og, cmp="RGB")
    plt.title(class_names[np.argmax(prediction)])
    plt.show()
