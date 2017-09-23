from model import get_detect_model
from common import CHARS
import numpy as np

weights_file = 'model_weights_fc1_7.h5'

def make_scaled_ims(im, min_shape):
    ratio = 1. / 2 ** 0.5
    shape = (im.shape[0] / ratio, im.shape[1] / ratio)

    result = []

    while True:
        shape = (int(shape[0] * ratio), int(shape[1] * ratio))
        if shape[0] < min_shape[0] or shape[1] < min_shape[1]:
            break
        result.append(cv2.resize(im, (shape[1], shape[0])))

    return np.array(result)

def detect(image):
    scaled_ims = make_scaled_ims(im, (64, 128))

    detect_model = get_detect_model(weights_file)
    y_vals = detect_model.predict_on_batch(scaled_ims)

    for i, (scaled_im, y_val) in enumerate(scaled_ims, y_vals):
        print(y_val['conv_presence_indicator'])

if __name__ == '__main__':
    im = cv2.imread(sys.argv[1])
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.
