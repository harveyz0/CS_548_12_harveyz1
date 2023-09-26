#!/usr/bin/python3

'''
Email: harveyz1@sunypoly.edu
Author: Zachary Harvey
Exercises from CS548 12 Slide deck 3
'''


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Flatten, Conv2D, MaxPooling2D, Add, Lambda
from tensorflow.keras import utils, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
import numpy as np
import cv2


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    print("x_train: ", x_train.shape)
    print("x_test: ", x_test.shape)
    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)
    print("x_train: ", x_train.shape)
    print("x_test: ", x_test.shape)

    model = build_model(shape=x_train.shape[1:])

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)

    model.fit(x_train, y_train, batch_size=32, epochs=10, callbacks=[tb_callback])

    train_scores = model.evaluate(x_train, y_train, batch_size=128)
    test_scores = model.evaluate(x_test, y_test, batch_size=128)

    print("Train: ", train_scores)
    print("Test: ", test_scores)


def build_sequential_model(shape=(28, 28, 1)):
    model = Sequential()
    model.add(InputLayer(input_shape=shape))
    model.add(Conv2D(32, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(64, kernel_size=3, activation="relu"))
    model.add(Conv2D(64, kernel_size=3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model


def build_model(shape=(28, 28, 1)):
    my_input = Input(shape=shape)
    x = Conv2D(32, kernel_size=3, padding="same", activation="relu")(my_input)
    x = Conv2D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = MaxPooling2D(2)(x)
    alt_x = Dense(64)(x)
    x = Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = Add()([x, alt_x])
    x = MaxPooling2D(2)(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    #my_output = Dense(10, activation='softmax')(x)
    #model = Model(inputs=my_input, outputs=my_output)
    base_model = VGG19(weights="imagenet", include_top=False)
    for layer in base_model.layers:
        layer.trainable = False

    true_input = Input(shape=shape)
    resized = Lambda(input_shape=shape, function=lambda images, h=224, w=224: tf.image.resize(images, [h, w]))(true_input)

    x = base_model(resized)
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(10, activation="softmax")(x)

    model = Model(inputs=true_input, outputs=x)
    model.summary()

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model


def preprocess_images(x: np.ndarray):
    x = x.astype("float32")
    x /= 255.0
    if len(x.shape) <= 3:
        x = add_dims(x)
    return x


def add_dims(data, axis=-1):
    return np.expand_dims(data, axis=axis)


def normalize(data, add_channel=True):
    data = data.astype("float32")
    data /= 255.0
    data -= 0.5
    data *= 2.0
    if add_channel:
        data = add_dims(data)
    return data


def prepare_image_data(data, add_channel=True):
    normalize(data, add_channel)
    data = utils.to_categorical(data, num_classes=10)


def show_images(data):
    for i in range(5):
        show_image(data[i])


def show_image(image):
    cv2.imshow("IMAGESSSSS", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    load_mnist()


if __name__ == '__main__':
    main()
