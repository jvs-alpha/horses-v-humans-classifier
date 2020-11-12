import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


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
    images, label = loadtest()
    class_names = ["horse", "human"]
    for i in range(20):
        plt.imshow(images[i])
        plt.title(class_names[label[i]])
        plt.show(block=False)
        plt.pause(3)
        plt.close()
