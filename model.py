from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Input

def convolutional_layers():
    img_input = Input(shape=(None, None, 1))

    # 1 layer
    x = Conv2D(48, (5, 5), padding="same")(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 2 layer
    x = Conv2D(64, (5, 5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)

    # 3 layer
    x = Conv2D(128, (5, 5), padding="same")(x)
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

    pass

def get_detect_model():
    """
    The same as training model, except it acts on arbitary sized image and
    slides the 128x64 window across the image in 8x8 strides
    """
    pass
