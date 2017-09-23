from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import BatchNormalization

from multi_gpu import make_parallel

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
    # x = Dense(2048, activation='relu', name='fc_2')(x)

    # output = Dense(1 + 8 * len(common.CHARS), activation='softmax', name='chars')(x)

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

def get_detect_model(trained_weights):
    """
    The same as training model, except it acts on arbitary sized image and
    slides the 128x64 window across the image in 8x8 strides

    not usable!
    TODO: restructurize model and load trained weights!!!
    """
    img_input = Input(shape=(None, None, 1))
    x = convolutional_layers(img_input)

    x = Conv2D(2048, (4, 16), activation='relu', name='conv_fc_1')(x)
    # x = Conv2D(2048, (8, 32), padding="valid", strides=(1, 1), activation='relu', name='conv_fc_2')(x)

    # x = Conv2D(4096, (4, 8), padding="valid", strides=(1, 1), activation='relu', name='conv_fc_1')(x)

    presence_indicator = Conv2D(1, (1, 1), activation='sigmoid', name='conv_presence_indicator')(x)
    char_1 = Conv2D(len(common.CHARS), (1, 1), activation='softmax', name='conv_char_1')(x)
    char_2 = Conv2D(len(common.CHARS), (1, 1), activation='softmax', name='conv_char_2')(x)
    char_3 = Conv2D(len(common.CHARS), (1, 1), activation='softmax', name='conv_char_3')(x)
    char_4 = Conv2D(len(common.CHARS), (1, 1), activation='softmax', name='conv_char_4')(x)
    char_5 = Conv2D(len(common.CHARS), (1, 1), activation='softmax', name='conv_char_5')(x)
    char_6 = Conv2D(len(common.CHARS), (1, 1), activation='softmax', name='conv_char_6')(x)
    char_7 = Conv2D(len(common.CHARS), (1, 1), activation='softmax', name='conv_char_7')(x)
    char_8 = Conv2D(len(common.CHARS), (1, 1), activation='softmax', name='conv_char_8')(x)

    # model = Model(inputs=img_input, outputs=[presence_indicator, encoded_chars])

    # return model
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

if __name__ == '__main__':
    training_model = get_training_model()
    training_model.summary()

    detect_model = get_detect_model()
    detect_model.summary()
