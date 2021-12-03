import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses, Model
import cv2
import csv
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
import os.path

def get_train_data(directory):
    x_train = []

    t = trash_fill['filename']
    i = 0
    # for item in trash_fill['filename'][0:1500]:
    #   x_train.append(cv2.imread('/content/drive/MyDrive/trash_can/' + item))
    #   i += 1
    #   print(i)

    x_train += [cv2.imread(directory + item) for item in trash_fill['filename'][:]]

    x_train = np.array(x_train)
    y_train = []
    for fill in trash_fill['fill'][:]:
        y_train.append(fill)
    y_train = np.array(y_train)
    print(x_train.shape, y_train.shape)
    return (x_train, y_train)

def build_resNet():
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = layers.Flatten()(base_model.output)
    x = layers.Dense(100, activation='relu')(x)
    predictions = layers.Dense(2, activation='softmax')(x)
    return (base_model, predictions)
def compile_network():
    head_model = Model(inputs=base_model.input, outputs=predictions)
    head_model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    return head_model

def fit():
    weights_file = 'resNet125.h5'
    if (os.path.exists(weights_file)):
        model.load_weights(weights_file)
    else:
        history = model.fit(x_train, y_train, batch_size=64, epochs=2)
        model.save(weights_file)

def res_graphic(history):
    fig, axs = plt.subplots(2, 1, figsize=(15, 15))
    axs[0].plot(history.history['loss'])
    # axs[0].plot(history.history['val_loss'])
    axs[0].title.set_text('Training Loss vs Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend(['Train', 'Val'])
    axs[1].plot(history.history['accuracy'])
    # axs[1].plot(history.history['val_accuracy'])
    axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend(['Train', 'Val'])

def predict(model):
    x_train = []
    num = 79
    img = cv2.imread('/content/drive/MyDrive/trash_can/' + trash_fill['filename'][num - 1])
    expected = trash_fill['fill'][num - 1]
    x_train.append(img)
    actual = model.predict(np.array(x_train))
    print(expected)
    print(actual)
    # cv2.cv2_imshow(img)


if __name__ == "__main__":
    trash_fill = pd.read_csv('./trash_fill.csv')
    x_train, y_train = get_train_data('./trash_can/')

    base_model, predictions = build_resNet()
    model = compile_network()
    history = fit()
    res_graphic(history)


