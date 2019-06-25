"""
modified keras code from the following sources:
https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd
"""
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense, Dropout
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam

import config as CFG
from datagen import DataGen
from graph import Graph

import cv2
import os
import numpy as np
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import pandas as pd
from math import ceil


def keras_network(train_data, train_labels, test_data, test_labels, batch):
    """
    Creates a neural network using the Keras library with Tensorflow backend
    :param train_data: sliced training data
    :param train_labels: actual labels for training data
    :param test_data: sliced testing data
    :param test_labels: actual labels for test data
    :param batch: batch size for gradient descent
    """
    training_accuracy = []
    training_loss = []
    testing_accuracy = []
    testing_loss = []

    # preprocess images: augment data
    if CFG.AUGMENT_IMAGES:
        data_generation = ImageDataGenerator(
            # Randomly apply the following parameters to the images
            width_shift_range=0.1,    # shift images horizontally
            height_shift_range=0.1,   # shift images vertically
            shear_range=0.1,          # shear images
            fill_mode='nearest',      # set mode for filling points outside the input boundaries
            zoom_range=[0.95, 1.05],  # set a zoom range randomly applied to images
            horizontal_flip=True)     # flip images horizontally

        data_generation.fit(train_data)
        steps_per_epoch = len(train_data) // batch
        print(f'steps_per_epoch = {steps_per_epoch}')

    classifier = Sequential()

    # build model
    print(f'build model')
    # input layer
    classifier.add(Convolution2D(32, (3, 3), input_shape=train_data.shape[1:], activation=CFG.ACTIVATE_FN[0]))
    classifier.add(AveragePooling2D(pool_size=(2, 2)))

    if CFG.DROPOUT:
        classifier.add(Dropout(0.2))

    # added layer
    classifier.add(Convolution2D(128, (3, 3), activation=CFG.ACTIVATE_FN[1]))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # layer
    classifier.add(Flatten())
    classifier.add(Dense(activation=CFG.ACTIVATE_FN[2], units=128))

    # output layer
    classifier.add(Dense(activation=CFG.ACTIVATE_FN[3], units=CFG.TOTAL_LABELS))

    # compile model
    print(f'compile model')
    optimizer = adam(lr=CFG.ETA, beta_1=CFG.BETA1, beta_2=CFG.BETA2)
    classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # train the model
    print(f'train model')
    for e in range(CFG.EPOCHS):
        print(f'===================== EPOCH {e} =====================')

        if CFG.AUGMENT_IMAGES:
            tr_results = classifier.fit_generator(data_generation.flow(train_data, train_labels, batch_size=batch),
                                                               epochs=1, steps_per_epoch=steps_per_epoch)
        else:
            tr_results = classifier.fit(train_data, train_labels, batch_size=batch, epochs=1)

        classifier.summary()

        # run the classifier on the test data
        epoch_accuracy = classifier.evaluate(test_data, test_labels, batch_size=batch)
        print(f'=============== TEST RESULTS ===============')
        print(f'evaulation classifier score: {epoch_accuracy}')
        print("%s: %.2f%%" % (classifier.metrics_names[1], epoch_accuracy[1] * 100))
        print(f'============================================')
        testing_accuracy.append(epoch_accuracy[1])
        testing_loss.append(epoch_accuracy[0])
        training_accuracy.append(tr_results.history['categorical_accuracy'])
        training_loss.append(tr_results.history['loss'])
    print(f'=====================================================')

    path = CFG.GRAPHS_DIR + 'summary.txt'
    with open(path, 'w') as file:
        with redirect_stdout(file):
            classifier.summary()

    return testing_accuracy, testing_loss, parse_functions(CFG.ACTIVATE_FN), training_accuracy, training_loss


def parse_functions(activation_fns: [str]):
    """
    Parses all of the activation functions so that they can be used as descriptors
    :param activation_fns: all activation functions that were used
    :return: a string of activation functions
    """
    total = len(activation_fns)
    if total < 2:
        if total < 1:
            return str(activation_fns)
        return activation_fns[1]

    fn_strings = ""
    for fn in range(total):
        fn_strings += activation_fns[fn]
        if fn != total - 1:
            fn_strings += "-"

    return fn_strings


def visualization(count, training_size, training_accuracy, training_cost, test_size, test_accuracy, test_cost,
                  function_name, batch):

    # create graph objects
    graph = Graph(count, training_size, training_accuracy, test_size, test_accuracy, function_name,
                           batch, "Accuracy Plot")

    # plot accuracy
    graph.plot_accuracy()

    # generate tabular data
    graph.tabular_data()

    # generate training cost plot
    if CFG.PLOT_COST:
        graph.plot_cost(training_cost.ravel(), test_cost.ravel())


def load_data() -> ((), ()):
    """
    Loads images of from defined dataset
    :return: lists of training data and testing data
    """
    # generate images from wavfiles
    if CFG.CONVERT_AUDIO:
        datagen = DataGen()
        datagen.convert_audio_into_images()

    row = CFG.ROW
    col = CFG.COL
    depth = CFG.DEPTH

    if CFG.MNIST_DATA:
        # use MNIST data
        # test data shape: (10000, 28, 28), labels: (10000, )
        # train data shape: (60000, 28, 28), labels: (60000, )
        # e.g. single data sample: (1, 28, 28), label: (1, )
        (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    else:
        # use custom data
        (train_data, train_labels), (test_data, test_labels), row, col, depth = custom_load()

    print(f'training_data: {train_data}')
    print(f'training_labels: {train_labels}')
    print(f'test data: {test_data}')
    print(f'test labels: {test_labels}')
    train_data = train_data.astype('float32') / 255
    test_data = test_data.astype('float32') / 255

    # resize to work with tensorflow
    training_data = train_data.reshape(train_data.shape[0], row, col, depth)
    testing_data = test_data.reshape(test_data.shape[0], row, col, depth)

    total_classes = CFG.TOTAL_LABELS
    training_labels = to_categorical(train_labels, total_classes)
    test_labels = to_categorical(test_labels, total_classes)

    return (training_data, training_labels), (testing_data, test_labels)


def custom_load() -> ((), (), int, int, int):
    """
    Manually loads custom data and preprocesses the data into training data and
    testing data
    :return: lists of training data and testing data, and original image row, column and depth
    """
    images = os.listdir(CFG.DATA_SOURCE)
    image_rgb = []
    image_labels = []

    for count, img in enumerate(images):

        img_sample = os.path.join(CFG.DATA_SOURCE, img)

        if os.path.isfile(img_sample):
            rgb_img = cv2.imread(img_sample, cv2.IMREAD_COLOR)

            if CFG.GENERATE_RESIZED_IMAGES:
                # resource for resizing images with cv2 while maintaining aspect ratio
                # https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
                width = int(rgb_img.shape[1] * CFG.SCALE_PERCENT)
                height = int(rgb_img.shape[0] * CFG.SCALE_PERCENT)
                rgb_img = cv2.resize(rgb_img, (width, height), interpolation=cv2.INTER_AREA)

            # Save a sample resized image
            if count == 0:
                cv2.imwrite('resized_sample.png', rgb_img)

            row, col, depth = rgb_img.shape
            print(f'REDUCED IMAGE SHAPE:')
            print(f'row: {row}, col: {col}, depth: {depth}')
            array_size = row * col * depth
            rgb_img.reshape(1, array_size)
            image_labels.append(label(img))
            image_rgb.append(rgb_img)

    rows = len(image_rgb)
    total_images = np.asarray(image_rgb).reshape((rows, array_size))
    total_labels = np.asarray(image_labels).reshape((rows, 1))

    # create full dataframe with labels
    data = np.concatenate((total_images, total_labels), axis=1)
    total_data, total_features = data.shape
    print(f'total data {total_data}, total features: {total_features}')

    # further randomize data
    np.random.shuffle(data)

    # split into training data and testing data
    split = 0
    if CFG.TRAIN_PERCENT != 0:
        split = ceil(total_data * CFG.TRAIN_PERCENT)
        if ceil == total_data:
            split = 0

    if split != 0:
        training_data = data[:split]
        test_data = data[split:]

    # TODO: handle if split = 0

    print(f'row: {row}, col: {col}, depth: {depth}')
    print(f'training_data: {training_data[:, :-1]}, training_labels {training_data[:, -1:].ravel()}')
    return (training_data[:, :-1], training_data[:, -1:].ravel()), \
           (test_data[:, :-1], test_data[:, -1:].ravel()), \
           row, col, depth


def label(filename: str) -> int:
    """
    Processes a filename in order to return the sample's label as an integer
    :param filename: filename of type string
    :return: label (e.g. if the filename contains a sample of audio 4, a 4 would be returned)
    """
    str_label = filename[:2]
    label = ord(str_label[1]) - 48
    label += (ord(str_label[0]) - 48) * 10
    return label


def main():

    experiment_count = 1

    # grab specified data set and preprocess it for network training
    (train_data, train_labels), (test_data, test_labels) = load_data()
    training_size, _, _, _ = train_data.shape
    test_size, _, _, _ = test_data.shape

    # use Keras to create Neural Network
    test_accuracy, test_loss, function_names, \
        training_accuracy, training_loss = keras_network(train_data, train_labels, test_data, test_labels,
                                                         CFG.BATCH_SIZE)

    # Test print of results
    print(f'====================== RESULTS ======================')
    print(f'\ntraining_accuracy: {training_accuracy}\n')
    print(f'training_loss: {training_loss}\n')
    print(f'test_accuracy: {test_accuracy}\n')
    print(f'test_loss: {test_loss}\n')
    print(f'=====================================================')

    training_accuracy = np.asarray(training_accuracy)
    training_loss = np.asarray(training_loss)

    test_accuracy = np.asarray(test_accuracy)
    test_loss = np.asarray(test_loss)

    training_accuracy = np.nan_to_num(training_accuracy)
    training_loss = np.nan_to_num(training_loss)

    test_accuracy = np.nan_to_num(test_accuracy)
    test_loss = np.nan_to_num(test_loss)

    training_accuracy *= 100
    test_accuracy *= 100

    # visualize various results in plots
    visualization(experiment_count, training_size, training_accuracy, training_loss, test_size, test_accuracy,
                  test_loss, function_names, CFG.BATCH_SIZE)

    experiment_count += 1


if __name__ == '__main__':
    main()
