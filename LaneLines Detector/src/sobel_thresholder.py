import cv2
import numpy as np

'''
Module to group the static methods transforming the images to binary format after thresholding
'''
THRES_RANGE = (0, 255)
DEFAULT_ABS_THRES = None
DEFAULT_SOBEL_KERNEL = None



def abs_sobel_threshold(img, orient='x', sobel_kernel=DEFAULT_SOBEL_KERNEL, thres=DEFAULT_ABS_THRES, is_gray=False):
    """Apply the absolute Sobel filter.
    Sobel operator detects gradients in x and y directions.
    """
    # Apply the following steps to img
    # 1) Convert to grayscale
    if is_gray is True:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        raise ValueError('orient must be "x" or "y".')
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    if thres is not None:
        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        sbinary = np.zeros_like(scaled_sobel)
        sbinary[(scaled_sobel >= thres[0]) & (scaled_sobel <= thres[1])] = 1
        # 6) Return this mask as binary_output image
        binary_output = sbinary
    else:
        binary_output = scaled_sobel
    return binary_output


