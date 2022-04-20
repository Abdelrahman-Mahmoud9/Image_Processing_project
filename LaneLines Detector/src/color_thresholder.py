import cv2
import numpy as np

COLOR_SPACES = ['RGB', 'HLS', 'HSV']


def convert_color(img, dest_color_space='HLS'):
    if dest_color_space == 'YCrCb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    elif dest_color_space == 'YUV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    elif dest_color_space == 'LUV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    elif dest_color_space == 'HLS':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    elif dest_color_space == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif dest_color_space == 'grayscale':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif dest_color_space == 'LAB':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    return img


def split_channels(img, color_space='HLS'):
    converted_image = convert_color(img, dest_color_space=color_space)
    image_as_np_array = converted_image.astype(np.float)
    ch1 = image_as_np_array[:, :, 0]
    ch2 = image_as_np_array[:, :, 1]
    ch3 = image_as_np_array[:, :, 2]
    return ch1, ch2, ch3


