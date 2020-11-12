from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import os
import random
import sys


def loadtest():
    train = []
    label = []
    trainfiles = os.listdir("validation")
    for i in trainfiles:
        if "horse" in i:
            name = 0
        else:
            name = 1
        with open(os.path.join("validation/", i), "rb") as img:
            trainimage = Image.open(img).convert("L")
            trainimage = np.asarray(trainimage)
            train.append(trainimage)
            label.append(name)
    return np.asarray(train), np.asarray(label)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Run python hvmtest.py <model> <type>")
        sys.exit(1)
    class_names = ["horse", "human"]
    test, t_label = loadtest()
    model = keras.models.load_model(sys.argv[1])
    test_loss, test_acc = model.evaluate(test,t_label)
    print("The test accuracy", test_acc)
    prediction = model.predict(test)
    for i in prediction:
        print(np.argmax(i))
    random.seed(3)
    if sys.argv[2] == "image":
        for i in range(20):
            index = random.randint(0, len(test)-1)
            plt.grid(False)
            plt.imshow(test[i],cmap="gray")
            plt.xlabel("Actual: " + class_names[t_label[index]])
            plt.title("Prediction " + class_names[np.argmax(prediction[index])])
            plt.show(block=False)
            plt.pause(3)
            plt.close()
    else:
        pass
        # for i in range(len(test)):
