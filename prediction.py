from PIL import Image
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def loadimage(filename):
    train = []
    with open(os.path.join("validation/", filename), "rb") as img:
        trainimage = Image.open(img).convert("L")
        trainimage = np.asarray(trainimage)
        train.append(trainimage)
    return np.asarray(train)


if __name__ == "__main__":
    og = loadimage("valhuman03-05.png")
    class_names = ["horse", "human"]
    model = keras.models.load_model("hvm_f1.h5")
    prediction = model.evaluate(og)
    plt.imshow(og[0], cmap="gray")
    plt.title(class_names[np.argmax(prediction)])
    plt.show()
