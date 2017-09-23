from model import get_detect_model
from common import CHARS
import numpy as np
import cv2
import sys

weights_file = 'model_weights_fc1_7.h5'
WINDOW_SHAPE = (64, 128)

def make_scaled_ims(im, min_shape):
    ratio = 1. / 2 ** 0.5
    shape = (im.shape[0] / ratio, im.shape[1] / ratio)

    while True:
        shape = (int(shape[0] * ratio), int(shape[1] * ratio))
        if shape[0] < min_shape[0] or shape[1] < min_shape[1]:
            break
        yield cv2.resize(im, (shape[1], shape[0]))

def detect(image):
    # print(image.shape)
    # im = image.reshape(1, image.shape[0], image.shape[1], 1)
    detect_model = get_detect_model(weights_file)
    scaled_ims = list(make_scaled_ims(im, WINDOW_SHAPE))
    y_vals = []
    for scaled_im in scaled_ims:
        val = detect_model.predict(scaled_im.reshape(1, scaled_im.shape[0], scaled_im.shape[1], 1), batch_size=1)
        y_vals.append(val)

    # y_val = detect_model.predict(im, batch_size=1)
    # print(len(y_val))
    # print(y_val[0].shape)
    # print(y_val[0][0][15][21][0])
    # print(y_val[1].shape)
    # print(y_val[2].shape)
    # print(y_val[3].shape)

    for i, (scaled_im, y_val) in enumerate(scaled_ims, y_vals):
        for window_coords in np.argwhere(y_val[0][0, :, :, 0] > 0.7):
            letter_probs = np.array([
                y_val[1][0, window_coords[0], window_coords[1], :]
                y_val[2][0, window_coords[0], window_coords[1], :]
                y_val[3][0, window_coords[0], window_coords[1], :]
                y_val[4][0, window_coords[0], window_coords[1], :]
                y_val[5][0, window_coords[0], window_coords[1], :]
                y_val[6][0, window_coords[0], window_coords[1], :]
                y_val[7][0, window_coords[0], window_coords[1], :]
                y_val[8][0, window_coords[0], window_coords[1], :]
            ])

            present_prob = y_val[0][0][window_coords[0]][window_coords[1]][0]

            img_scale = float(image.shape[0]) / scaled_im.shape[0]
            bbox_tl = window_coords * (16, 8) * img_scale
            bbox_size = np.array(WINDOW_SHAPE) * img_scale

            yield bbox_tl, bbox_tl + bbox_size, present_prob, letter_probs


def _overlaps(match1, match2):
    bbox_tl1, bbox_br1, _, _ = match1
    bbox_tl2, bbox_br2, _, _ = match2
    return (bbox_br1[0] > bbox_tl2[0] and
            bbox_br2[0] > bbox_tl1[0] and
            bbox_br1[1] > bbox_tl2[1] and
            bbox_br2[1] > bbox_tl1[1])


def _group_overlapping_rectangles(matches):
    matches = list(matches)
    num_groups = 0
    match_to_group = {}
    for idx1 in range(len(matches)):
        for idx2 in range(idx1):
            if _overlaps(matches[idx1], matches[idx2]):
                match_to_group[idx1] = match_to_group[idx2]
                break
        else:
            match_to_group[idx1] = num_groups
            num_groups += 1

    groups = collections.defaultdict(list)
    for idx, group in match_to_group.items():
        groups[group].append(matches[idx])

    return groups


def post_process(matches):
    """
    Take an iterable of matches as returned by `detect` and merge duplicates.

    Merging consists of two steps:
      - Finding sets of overlapping rectangles.
      - Finding the intersection of those sets, along with the code
        corresponding with the rectangle with the highest presence parameter.

    """
    groups = _group_overlapping_rectangles(matches)

    for group_matches in groups.values():
        mins = np.stack(np.array(m[0]) for m in group_matches)
        maxs = np.stack(np.array(m[1]) for m in group_matches)
        present_probs = np.array([m[2] for m in group_matches])
        letter_probs = np.stack(m[3] for m in group_matches)

        yield (np.max(mins, axis=0).flatten(),
               np.min(maxs, axis=0).flatten(),
               np.max(present_probs),
               letter_probs[np.argmax(present_probs)])

def letter_probs_to_code(letter_probs):
    return "".join(CHARS[i] for i in np.argmax(letter_probs, axis=1))

if __name__ == '__main__':
    im = cv2.imread(sys.argv[1])
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.

    for pt1, pt2, present_prob, letter_probs in post_process(detect(im_gray)):
        pt1 = tuple(reversed(map(int, pt1)))
        pt2 = tuple(reversed(map(int, pt2)))

        code = letter_probs_to_code(letter_probs)

        color = (0.0, 255.0, 0.0)
        cv2.rectangle(im, pt1, pt2, color)

        cv2.putText(im,
                    code,
                    pt1,
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (0, 0, 0),
                    thickness=5)

        cv2.putText(im,
                    code,
                    pt1,
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (255, 255, 255),
                    thickness=2)

    cv2.imwrite(sys.argv[2], im)
