import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import random


if __name__ == "__main__":
    train = keras.preprocessing.image_dataset_from_directory(
        "train",
        seed=123,
        image_size=(300,300),
        batch_size=345
    )
    class_names = train.class_names
    print(train.images)
    # model = keras.Sequential([
    #     keras.layers.Flatten(input_shape=(300,300)),
    #     keras.layers.Dense(130,activation="relu"),
    #     # keras.layers.Dense(80,activation="relu"),
    #     keras.layers.Dense(2,activation="softmax")
    #     ])
    # model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    # model.fit(train,label,batch_size=3,epochs=8)
    # test, t_label = loadtest()
    # test_loss, test_acc = model.evaluate(test,t_label)
    # print("The test accuracy", test_acc)
    # model.save("hvm.h5")
    # prediction = model.predict(test)
    # for i in range(20):
    #     plt.grid(False)
    #     plt.imshow(test[i],cmap="gray")
    #     plt.xlabel("Actual: " + class_names[t_label[i]])
    #     plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    #     plt.show(block=False)
    #     plt.pause(3)
    #     plt.close()
