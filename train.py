import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import model as models
import common
from keras.callbacks import ModelCheckpoint
from gen import generate_ims
import numpy as np
import itertools
from keras.models import load_model
from multi_gpu import make_parallel

weights_file = 'model_weights_fc2_single_output.h5'
batch_size = 128

steps_per_epoch = 250
num_epochs = 5000
validation_steps = 4

# def code_to_vec(code):
#     def char_to_vec(c):
#         y = np.zeros((len(common.CHARS),))
#         y[common.CHARS.index(c)] = 1.0
#         return y
#
#     c = np.vstack([char_to_vec(c) for c in code])
#
#     return c.flatten()
#
# def unzip(b):
#     xs, y1s, y2s = zip(*b)
#     xs = np.array(xs)
#     y1s = np.array(y1s)
#     y2s = np.array(y2s)
#     return xs, {'presence_indicator':y1s, 'encoded_chars':y2s}

def code_to_vec(p, code):
    def char_to_vec(c):
        y = np.zeros((len(common.CHARS),))
        y[common.CHARS.index(c)] = 1.0
        return y

    c = np.vstack([char_to_vec(c) for c in code])

    return np.concatenate([[1. if p else 0], c.flatten()])

def unzip(b):
    xs, ys = zip(*b)
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys

def read_batches(batch_size):
    g = generate_ims()
    def gen_vecs():
        for im, c, p in itertools.islice(g, batch_size):
            yield im.reshape(64, 128, 1), code_to_vec(p, c)

    while True:
        yield unzip(gen_vecs())

training_model = models.get_training_model()
training_model.load_weights(weights_file, by_name=True)
# training_model.compile(
#     loss={'presence_indicator':'binary_crossentropy', 'encoded_chars':'categorical_crossentropy'},
#     optimizer='adam',
#     metrics={'presence_indicator':'binary_accuracy', 'encoded_chars':'categorical_accuracy'})

training_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

print('\nStarting training...\n')
training_model.fit_generator(read_batches(batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,
    # validation_data=read_batches(batch_size),
    # validation_steps=validation_steps,
    callbacks=[
        ModelCheckpoint(weights_file, save_best_only=True, monitor='loss')
    ])
