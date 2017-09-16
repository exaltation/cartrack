from keras.models import Model
from keras.layers import Input
from keras.layers import Dense

img_input = Input(shape=(8, 32, 128))

x = Dense(2048, name='fc1')(img_input)
m = Model(inputs=img_input, outputs=x)
