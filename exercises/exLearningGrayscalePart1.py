#!/usr/bin/python
'''
Email: harveyz1@sunypoly.edu
Author: Zachary Harvey
Exercises from CS548 12 Slide deck 1 Learning Grey scale
'''

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
import cv2
import numpy as np
from sys import argv, exit
import os


def build_model():
    input_node = Input(shape=(None, None, 3))
    filter_layer = Dense(1, use_bias=False)
    output_node = filter_layer(input_node)
    model = Model(inputs=input_node, outputs=output_node)
    model.compile(optimizer="adam", loss="mse", metrics=['mse'])
    return filter_layer, model


def load_image(file_image="./test.jpg"):
    if not os.path.exists(file_image):
        print(f"File does not exists {file_image}")
        exit(1)
    image = cv2.imread(file_image)

    grey_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_scale = np.expand_dims(grey_scale, axis=-1)

    batch_input = np.expand_dims(image, axis=0)
    batch_output = np.expand_dims(gray_scale, axis=0)

    return batch_input.astype("float32") / 255.0, \
        batch_output.astype("float32") / 255.0


def fit_image(file_image="./test.jpg"):
    filter_layer, model = build_model()
    batch_input, batch_output = load_image(file_image)
    model.fit(batch_input, batch_output, epochs=100, batch_size=1)
    print("WEIGHTS: \n", filter_layer.weights[0].numpy())

    pred_image = model.predict(batch_input)
    pred_image = np.squeeze(pred_image, axis=0)
    pred_image *= 255.0
    pred_image = cv2.convertScaleAbs(pred_image)
    show_image(pred_image)


def show_image(image):
    cv2.imshow("IMAGESSSSS", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(args):
    file_image = args[1] if len(args) > 1 else ""
    if file_image == "":
        file_image = "./test.jpg"
    fit_image(file_image)


if __name__ == '__main__':
    main(argv)
