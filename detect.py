from model import get_detect_model
from common import CHARS
import numpy as np
import cv2
import sys

weights_file = 'model_weights_fc1_7.h5'

def detect(image):
    print(image.shape)
    im = image.reshape(1, image.shape[0], image.shape[1], 1)
    detect_model = get_detect_model(weights_file)
    y_val = detect_model.predict(im, batch_size=1)
    print(len(y_val))
    print(y_val[0].shape)
    print(y_val[0][0][15][21][0])
    print(y_val[1].shape)
    print(y_val[2].shape)
    print(y_val[3].shape)
    # for i, (scaled_im, y_val) in enumerate(scaled_ims, y_vals):
    #     print(y_val['conv_presence_indicator'])
    #     if i > 25:
    #         return

if __name__ == '__main__':
    im = cv2.imread(sys.argv[1])
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.

    detect(im_gray)
