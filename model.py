from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import BatchNormalization

import common

def convolutional_layers(img_input):
    # 1 layer
    x = Conv2D(48, (3, 3), padding='same', name='conv_1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 2 layer
    x = Conv2D(64, (3, 3), padding='same', name='conv_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 3 layer
    x = Conv2D(128, (3, 3), padding='same', name='conv_3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 4 layer
    x = Conv2D(256, (3, 3), padding='same', name='conv_4')(x)
    x = BatchNormalization()(x)
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
    encoded_chars = Dense(8 * len(common.CHARS), activation='softmax', name='encoded_chars')(x)

    return Model(inputs=img_input, outputs=[presence_indicator, encoded_chars])

def get_detect_model(trained_weights):
    """
    The same as training model, except it acts on arbitary sized image and
    slides the 128x64 window across the image in 8x8 strides
    """
    img_input = Input(shape=(None, None, 1))
    x = convolutional_layers(img_input)
    x = Conv2D(2048, (8, 32), padding="valid", strides=(1, 1), activation='relu', name='conv_fc_1')(x)
    presence_indicator = Conv2D(1, (1, 1), activation='sigmoid', name='conv_presence_indicator')(x)
    encoded_chars = Conv2D(8 * len(common.CHARS), (1, 1), activation='softmax', name='conv_encoded_chars')(x)

    model = Model(inputs=img_input, outputs=[presence_indicator, encoded_chars])

    return model

if __name__ == '__main__':
    training_model = get_training_model()
    training_model.summary()

    detect_model = get_detect_model()
    detect_model.summary()
