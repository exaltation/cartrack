from keras.layers import concatenate
from keras.layers.core import Lambda
from keras.models import Model
from tensorflow.python.client import device_lib
import tensorflow as tf


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def make_parallel(model, gpu_count=None):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # if gpu_count provided, use it
    # otherwise use all available gpus
    if gpu_count:
        gpus = ['/gpu:%d' % i for i in range(gpu_count)]
    else:
        gpus = get_available_gpus()
        gpu_count = len(gpus)

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i, device in enumerate(gpus):
        with tf.device(device):
            with tf.name_scope('tower_%d' % i):
                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice,
                                     output_shape=input_shape,
                                     arguments={'idx': i,
                                                'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(concatenate(outputs, axis=0))
        return Model(inputs=model.inputs, outputs=merged)
