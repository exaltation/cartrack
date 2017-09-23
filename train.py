import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import model as models
import common
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from gen import generate_ims
import numpy as np
import itertools
from keras.models import load_model
from multi_gpu import make_parallel

weights_file = 'model_weights_fc1_7.h5'
batch_size = 128

steps_per_epoch = 250
num_epochs = 500

def unzip(b):
    xs, y0s, y1s, y2s, y3s, y4s, y5s, y6s, y7s, y8s = zip(*b)
    xs = np.array(xs)
    y0s = np.array(y0s)
    y1s = np.array(y1s)
    y2s = np.array(y2s)
    y3s = np.array(y3s)
    y4s = np.array(y4s)
    y5s = np.array(y5s)
    y6s = np.array(y6s)
    y7s = np.array(y7s)
    y8s = np.array(y8s)
    return xs, {
        'presence_indicator': y0s,
        'char_1': y1s,
        'char_2': y2s,
        'char_3': y3s,
        'char_4': y4s,
        'char_5': y5s,
        'char_6': y6s,
        'char_7': y7s,
        'char_8': y8s,
    }

def code_to_vec(code):
    def char_to_vec(c):
        y = np.zeros((len(common.CHARS),))
        y[common.CHARS.index(c)] = 1.0
        return y

    c = np.vstack([char_to_vec(c) for c in code])

    return c


def read_batches(batch_size):
    g = generate_ims()
    def gen_vecs():
        for im, c, p in itertools.islice(g, batch_size):
            _p = 1. if p else 0
            _c = code_to_vec(c)

            yield im.reshape(64, 128, 1), _p, _c[0], _c[1], _c[2], _c[3], _c[4], _c[5], _c[6], _c[7]

    while True:
        yield unzip(gen_vecs())

training_model = models.get_training_model()
training_model.load_weights(weights_file, by_name=True)

training_model.compile(
    loss={
        'presence_indicator':'binary_crossentropy',
        'char_1':'categorical_crossentropy',
        'char_2':'categorical_crossentropy',
        'char_3':'categorical_crossentropy',
        'char_4':'categorical_crossentropy',
        'char_5':'categorical_crossentropy',
        'char_6':'categorical_crossentropy',
        'char_7':'categorical_crossentropy',
        'char_8':'categorical_crossentropy',
    },
    optimizer='adadelta',
    metrics={
        'presence_indicator':'binary_accuracy',
        'char_1':'categorical_accuracy',
        'char_2':'categorical_accuracy',
        'char_3':'categorical_accuracy',
        'char_4':'categorical_accuracy',
        'char_5':'categorical_accuracy',
        'char_6':'categorical_accuracy',
        'char_7':'categorical_accuracy',
        'char_8':'categorical_accuracy',
    })

print('\nStarting training...\n')
training_model.fit_generator(read_batches(batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,
    verbose=1,
    callbacks=[
        ModelCheckpoint(weights_file, save_best_only=True, monitor='loss'),
        ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=1e-6, cooldown=5, min_lr=5e-5)
    ])
