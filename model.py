from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import BatchNormalization

import numpy as np

import common

def convolutional_layers(img_input):
    # 1 layer
    x = Conv2D(48, (7, 7), padding='same', name='conv_1')(img_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 2 layer
    x = Conv2D(64, (5, 5), padding='same', name='conv_2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)

    # 3 layer
    x = Conv2D(128, (5, 5), padding='same', name='conv_3')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 4 layer
    x = Conv2D(256, (3, 3), padding='same', name='conv_4')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    return x

def get_training_model():
    """
    Training model acts on a batch of 128x64 windows and outputs a
    1 + 8 * len(CHARS) vector
    """
    img_input = Input(shape=(64, 128, 1))
    x = convolutional_layers(img_input)

    x = Flatten()(x)
    x = Dense(2048, activation='relu', name='fc_1')(x)

    presence_indicator = Dense(1, activation='sigmoid', name='presence_indicator')(x)
    char_1 = Dense(len(common.CHARS), activation='softmax', name='char_1')(x)
    char_2 = Dense(len(common.CHARS), activation='softmax', name='char_2')(x)
    char_3 = Dense(len(common.CHARS), activation='softmax', name='char_3')(x)
    char_4 = Dense(len(common.CHARS), activation='softmax', name='char_4')(x)
    char_5 = Dense(len(common.CHARS), activation='softmax', name='char_5')(x)
    char_6 = Dense(len(common.CHARS), activation='softmax', name='char_6')(x)
    char_7 = Dense(len(common.CHARS), activation='softmax', name='char_7')(x)
    char_8 = Dense(len(common.CHARS), activation='softmax', name='char_8')(x)

    return Model(inputs=img_input, outputs=[
        presence_indicator,
        char_1,
        char_2,
        char_3,
        char_4,
        char_5,
        char_6,
        char_7,
        char_8,
    ])

def get_detect_model(trained_weights=False):
    """
    The same as training model, except it acts on arbitary sized image and
    slides the 128x64 window across the image with strides 16x8

    TODO: load trained weights
    """
    img_input = Input(shape=(None, None, 1))
    x = convolutional_layers(img_input)

    x = Conv2D(2048, (4, 16), activation='relu', name='conv_fc_1')(x)

    presence_indicator = Conv2D(1, (1, 1), activation='sigmoid', name='conv_presence_indicator')(x)
    char_1 = Conv2D(len(common.CHARS), (1, 1), activation='softmax', name='conv_char_1')(x)
    char_2 = Conv2D(len(common.CHARS), (1, 1), activation='softmax', name='conv_char_2')(x)
    char_3 = Conv2D(len(common.CHARS), (1, 1), activation='softmax', name='conv_char_3')(x)
    char_4 = Conv2D(len(common.CHARS), (1, 1), activation='softmax', name='conv_char_4')(x)
    char_5 = Conv2D(len(common.CHARS), (1, 1), activation='softmax', name='conv_char_5')(x)
    char_6 = Conv2D(len(common.CHARS), (1, 1), activation='softmax', name='conv_char_6')(x)
    char_7 = Conv2D(len(common.CHARS), (1, 1), activation='softmax', name='conv_char_7')(x)
    char_8 = Conv2D(len(common.CHARS), (1, 1), activation='softmax', name='conv_char_8')(x)

    m = Model(inputs=img_input, outputs=[
        presence_indicator,
        char_1,
        char_2,
        char_3,
        char_4,
        char_5,
        char_6,
        char_7,
        char_8,
    ])
    if trained_weights != False:
        m.load_weights(trained_weights, by_name=True)

        trained_model = load_model(trained_weights)
        fc_1 = trained_model.get_layer('fc_1').get_weights()
        presence_indicator = trained_model.get_layer('presence_indicator').get_weights()
        char_1 = trained_model.get_layer('char_1').get_weights()
        char_2 = trained_model.get_layer('char_2').get_weights()
        char_3 = trained_model.get_layer('char_3').get_weights()
        char_4 = trained_model.get_layer('char_4').get_weights()
        char_5 = trained_model.get_layer('char_5').get_weights()
        char_6 = trained_model.get_layer('char_6').get_weights()
        char_7 = trained_model.get_layer('char_7').get_weights()
        char_8 = trained_model.get_layer('char_8').get_weights()

        m.layers[13].set_weights([np.reshape(fc_1[0], m.get_layer('conv_fc_1').get_weights()[0].shape), fc_1[1]])
        m.layers[14].set_weights([np.reshape(presence_indicator[0], m.get_layer('conv_presence_indicator').get_weights()[0].shape), presence_indicator[1]])
        m.layers[15].set_weights([np.reshape(char_1[0], m.get_layer('conv_char_1').get_weights()[0].shape), char_1[1]])
        m.layers[16].set_weights([np.reshape(char_2[0], m.get_layer('conv_char_2').get_weights()[0].shape), char_2[1]])
        m.layers[17].set_weights([np.reshape(char_3[0], m.get_layer('conv_char_3').get_weights()[0].shape), char_3[1]])
        m.layers[18].set_weights([np.reshape(char_4[0], m.get_layer('conv_char_4').get_weights()[0].shape), char_4[1]])
        m.layers[19].set_weights([np.reshape(char_5[0], m.get_layer('conv_char_5').get_weights()[0].shape), char_5[1]])
        m.layers[20].set_weights([np.reshape(char_6[0], m.get_layer('conv_char_6').get_weights()[0].shape), char_6[1]])
        m.layers[21].set_weights([np.reshape(char_7[0], m.get_layer('conv_char_7').get_weights()[0].shape), char_7[1]])
        m.layers[22].set_weights([np.reshape(char_8[0], m.get_layer('conv_char_8').get_weights()[0].shape), char_8[1]])

    return m

if __name__ == '__main__':
    training_model = get_training_model()
    training_model.summary()

    detect_model = get_detect_model()
    detect_model.summary()
