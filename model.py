from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import BatchNormalization

import common
# from keras.layers import Dropout

def convolutional_layers():
    img_input = Input(shape=(None, None, 1))

    # 1 layer
    x = Conv2D(48, (5, 5), padding='same', name='conv_1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 2 layer
    x = Conv2D(64, (5, 5), padding='same', name='conv_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)

    # 3 layer
    x = Conv2D(128, (5, 5), padding='same', name='conv_3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # if input size is 128x64, output size will be 8x32x128
    return x

def get_training_model():
    """
    Training model acts on a batch of 128x64 windows and outputs a
    1 + 8 * len(CHARS) vector
    """
    x = convolutional_layers()
    x = Dense(2048, activation='relu', name='fc_1')(x)
    presence_idicator = Dense(1, activation='sigmoid', name='presence_idicator')(x)
    encoded_chars = Dense(8 * len(common.CHARS), activation='softmax', name='encoded_chars')(x)

    return Model(inputs=x, outputs=[presence_idicator, encoded_chars])

def get_detect_model():
    """
    The same as training model, except it acts on arbitary sized image and
    slides the 128x64 window across the image in 8x8 strides
    """
    x = convolutional_layers()
    x = Conv2D(2048, (8, 32), padding="same", strides=(8, 8), activation='relu', name='fc_1')(x)
    presence_idicator = Conv2D(1, (1, 1), activation='sigmoid', name='presence_idicator')(x)
    encoded_chars = Conv2D(8 * len(common.CHARS), (1, 1), activation='softmax', name='encoded_chars')(x)

    return Model(inputs=x, outputs=[presence_idicator, encoded_chars])
