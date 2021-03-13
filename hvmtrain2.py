import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
    train = keras.preprocessing.image_dataset_from_directory(
        "train",
        seed=123,
        image_size=(300,300),
        batch_size=3
    )
    test = keras.preprocessing.image_dataset_from_directory(
        "validation",
        seed=123,
        image_size=(300,300),
        batch_size=3
    )
    # Normalization layer for converting
    normalization_layer = keras.layers.experimental.preprocessing.Rescaling(1./255)
    # Normalization of train data
    # The x is the images data and the y is the label
    normalized_ds = train.map(lambda x, y: (normalization_layer(x), y))
    image_train, labels_train = next(iter(normalized_ds))
    # Normalization of test data
    normalized_ds = test.map(lambda x, y: (normalization_layer(x), y))
    image_test, labels_test = next(iter(normalized_ds))
    # THis is without normalization
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(300,300,3)),
        keras.layers.Dense(130,activation="relu"),
        keras.layers.Dense(80,activation="relu"),
        keras.layers.Dense(2,activation="softmax")
        ])
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    history = model.fit(train,validation_data=test,batch_size=3,epochs=4)
    test_loss, test_acc = model.evaluate(image_test,labels_test)
    print("The test accuracy", test_acc)
    model.save("hvm.h5")
