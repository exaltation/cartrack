import model as models
import common
from keras.callbacks import ModelCheckpoint
from gen import generate_ims
import numpy as np
import itertools

weights_file = 'model_weights.h5'
batch_size = 50

steps_per_epoch = 300
num_epochs = 300
validation_steps = 20

def code_to_vec(p, code):
    def char_to_vec(c):
        y = np.zeros((len(common.CHARS),))
        y[common.CHARS.index(c)] = 1.0
        return y

    c = np.vstack([char_to_vec(c) for c in code])

    return np.concatenate([(1. if p else 0), c.flatten()])

def unzip(b):
    xs, ys = zip(*b)
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys

def read_batches(batch_size):
    g = generate_ims()
    def gen_vecs():
        for im, c, p in itertools.islice(g, batch_size):
            yield im, code_to_vec(p, c)

    while True:
        yield unzip(gen_vecs())

# def data_generator():
#     g = generate_ims()
#     while True:
#         _inputs = []
#         _targets = []
#
#         for i in xrange(batch_size):
#             inputs, code, p =
#             targets = code_to_vec(p, code)
#
#             _inputs.append(inputs)
#             _targets.append(targets)
#
#         yield np.array(_inputs), np.array(_targets)

training_model = models.get_training_model()
training_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics={'presence_idicator':'binary_accuracy', 'encoded_chars':'categorical_accuracy'})

print('\nStarting training...\n')
training_model.fit_generator(read_batches(batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,
    validation_data=read_batches(batch_size),
    validation_steps=validation_steps,
    callbacks=[
        ModelCheckpoint(weights_file, save_best_only=True)
    ])
