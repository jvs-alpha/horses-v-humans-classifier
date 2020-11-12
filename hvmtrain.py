import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random


def loadtrain():
    train = []
    label = []
    trainfiles = os.listdir("train")
    for i in trainfiles:
        if "horse" in i:
            name = 0
        else:
            name = 1
        with open(os.path.join("train/", i), "rb") as img:
            trainimage = Image.open(img).convert("L")
            trainimage = np.asarray(trainimage)
            train.append(trainimage)
            label.append(name)
    return train, label

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
    btrain, blabel = loadtrain()
    train = []
    label = []
    while len(btrain) != 0 and len(blabel) != 0:
        random.seed(0)
        index = random.randint(0, len(btrain)-1)
        train.append(btrain.pop(index))
        label.append(blabel.pop(index))
    train = np.asarray(train)
    label = np.asarray(label)
    train = train/225.0
    print(train[0])
    # plt.imshow(train[0],cmap="gray")
    # plt.title(label[0])
    # plt.show()
    class_names = ["horse", "human"]
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(300,300)),
        keras.layers.Dense(130,activation="relu"),
        # keras.layers.Dense(80,activation="relu"),
        keras.layers.Dense(2,activation="softmax")
        ])
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    model.fit(train,label,batch_size=3,epochs=15)
    test, t_label = loadtest()
    test_loss, test_acc = model.evaluate(test,t_label)
    print("The test accuracy", test_acc)
    model.save("hvm.h5")
    prediction = model.predict(test)
    for i in range(20):
        plt.grid(False)
        plt.imshow(test[i],cmap="gray")
        plt.xlabel("Actual: " + class_names[t_label[i]])
        plt.title("Prediction " + class_names[np.argmax(prediction[i])])
        plt.show()
