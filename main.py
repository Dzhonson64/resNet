import json

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses, Model
import cv2
import numpy as np
import pandas as pd
from kafka import KafkaConsumer
import os.path
from kafka import KafkaProducer
from kafka.errors import KafkaError

def get_train_data(directory):
    x_train = []
    x_train += [cv2.imread(directory + item) for item in trash_fill['filename'][:]]

    x_train = np.array(x_train)
    y_train = []
    for fill in trash_fill['fill'][:]:
        y_train.append(fill)
    y_train = np.array(y_train)
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

def fit(model):
    weights_file = 'resNet.h5'
    if (os.path.exists(weights_file)):
        model.load_weights(weights_file)
    else:
        history = model.fit(x_train, y_train, batch_size=64, epochs=2)
        model.save(weights_file)
    return model

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

def predict(model, trash_fill):
    label = ["Незаполнен", "Заполнен"]
    x_train = []

    num = np.random.randint(1, 1800)
    img = cv2.imread('./trash_can/' + trash_fill['filename'][num - 1])
    expected_fill = trash_fill['fill'][num - 1]
    expected_filename = trash_fill['filename'][num - 1]
    x_train.append(img)
    actual = model.predict(np.array(x_train))
    print("expected_file_name=" + str(expected_filename), " expected_fill=" + str(label[expected_fill]), " actual_data=" + str(actual), "arg_max_actual=" + str(label[np.argmax(actual)]))
    return (num, actual[0])

def kafka_producer(img_num, prediction):

    producer = KafkaProducer(bootstrap_servers=['localhost:9092'], value_serializer=lambda m: json.dumps(m).encode('utf8'))
    # Asynchronous by default
    future = producer.send('detect-trash', {'prediction_unfilled': str(prediction[0]), 'prediction_filled': str(prediction[1]), 'image': img_num})

    # Block for 'synchronous' sends
    try:
        record_metadata = future.get(timeout=10)
        # Successful result returns assigned partition and offset
        print(record_metadata.topic)
        print(record_metadata.partition)
        print(record_metadata.offset)

    except KafkaError:
        # Decide what to do if produce request failed...
        print("KafkaError")
        pass


    # produce json messages



    def on_send_success(record_metadata):
        print(record_metadata.topic)
        print(record_metadata.partition)
        print(record_metadata.offset)

    def on_send_error(excp):
        print(excp)
        # handle exceptionz
    producer.flush()

def kafka_consumer(model, trash_fill):

    consumer = KafkaConsumer('topic-camera',
                             group_id='group1',
                             bootstrap_servers=['localhost:9092'],
                             auto_offset_reset="earliest")
    for message in consumer:
        img_num, prediction = predict(model, trash_fill)
        kafka_producer(img_num, prediction)
        # message value and key are raw bytes -- decode if necessary!
        # e.g., for unicode: `message.value.decode('utf-8')`
        print("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
                                             message.offset, message.key,
                                             message.value))

if __name__ == "__main__":
    # To consume latest messages and auto-commit offsets
    trash_fill = pd.read_csv('./trash_fill.csv')
    x_train, y_train = get_train_data('./trash_can/')

    base_model, predictions = build_resNet()
    model = compile_network()
    model = fit(model)
    kafka_consumer(model, trash_fill)




