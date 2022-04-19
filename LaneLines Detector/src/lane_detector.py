import numpy as np
import matplotlib.pyplot as plt
import cv2

IMG_WIDTH = 1280
IMG_HEIGHT = 720
LANE_WIDTH_PX = 640
YM_PER_PX = 30 / IMG_HEIGHT  # meters per pixel in y dimension
XM_PER_PX = 3.7 / LANE_WIDTH_PX  # meters per pixel in x dimension


def get_histogram(img):
    histogram = np.sum(img[int(img.shape[0] / 2):, :], axis=0)
    return histogram


def find_lane_lines(binary_warped, visualize=False, nwindows=9):
    """
    Find left and right lane lines in the provided binary warped image by initially
    using a histogram to find the lanes line in the bottom half of the image and then
    using a sliding window techniqe to iteratively move up and finf the next part of 
    the lane lines.
    Returns the points arrays to draw each of the left and right lanes as a 2nd order
    polynonial as well as the polynomial coefficients.
    
    :param binary_warped:
        Binary image that has already been warped (perspective transformed).
    
    :param visualize:
        Draw the sliding windows onto an output image.
    
    :returns:
        left_fitx - x values for plot of left lane
        right_fitx - x values for plot of right lane
        ploty - y values for plots of left and right lanes
        left_fit - 2nd order polynomial coefficients from np.polyfit for left lane
        right_fit - 2nd order polynomial coefficients from np.polyfit for right lane
    """
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    if visualize:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    return left_fitx, right_fitx, ploty, left_fit, right_fit


def find_lane_lines2(binary_warped, visualize=False, nwindows=9):
    """
    Find left and right lane lines in the provided binary warped image by initially
    using a histogram to find the lanes line in the bottom half of the image and then
    using a sliding window techniqe to iteratively move up and finf the next part of 
    the lane lines.
    Returns the points arrays to draw each of the left and right lanes as a 2nd order
    polynonial as well as the polynomial coefficients.
    
    :param binary_warped:
        Binary image that has already been warped (perspective transformed).
    
    :param visualize:
        Draw the sliding windows onto an output image.
    
    :returns:
        left_fitx - x values for plot of left lane
        right_fitx - x values for plot of right lane
        ploty - y values for plots of left and right lanes
        left_fit - 2nd order polynomial coefficients from np.polyfit for left lane
        right_fit - 2nd order polynomial coefficients from np.polyfit for right lane
    """
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        # cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        # cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit


def get_curvature_radius(leftx, rightx, ploty, xm_per_pix=XM_PER_PX, ym_per_pix=YM_PER_PX):
    """Calculate radius of curvature from pixels of left and right lanes in a image.
    Args:
        leftx, rightx, ploty (arrays): points on the lane
        xm_per_pix (float): Conversion in x from pixels space to merters
        ym_per_pix (float): Conversion in y from pixels space to merters
    Returns (float):
        Average of radius of left and right lane curvature in meters
    """
    # Fit new polynomials to x,y in world space
    y_eval = np.max(ploty)
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radius of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    center_curverad = np.average([left_curverad, right_curverad])

    # Now our radius of curvature is in meters
    return center_curverad, left_curverad, right_curverad


def show_inside_lane(undist_img, binary_warped, Minv, left_fitx, right_fitx, ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist_img.shape[1], undist_img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)  # image should be undistorted version
    return result


def dist_from_center2(img, left_fit, right_fit, xm_per_pix=XM_PER_PX):
    img_size = (img.shape[1], img.shape[0])

    left_intcpt = left_fit[0] * img_size[1] ** 2 + left_fit[1] * img_size[1] + left_fit[2]
    right_intcpt = right_fit[0] * img_size[1] ** 2 + right_fit[1] * img_size[1] + right_fit[2]
    lane_mid = (left_intcpt + right_intcpt) / 2.0
    car_off = (lane_mid - img_size[0] / 2.0) * xm_per_pix

    return car_off


def dist_from_center(left_fitx, right_fitx, img_width=IMG_WIDTH, xm_per_pix=XM_PER_PX):
    # Calculate distance from center
    # x position of left line at y = 720
    left_x = left_fitx[-1]
    right_x = right_fitx[-1]
    center_x = left_x + ((right_x - left_x) / 2)
    return ((img_width / 2) - center_x) * xm_per_pix


def generate_plot(img, lfit, rfit):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = lfit[0] * ploty ** 2 + lfit[1] * ploty + lfit[2]
    right_fitx = rfit[0] * ploty ** 2 + rfit[1] * ploty + rfit[2]
    return left_fitx, ploty, right_fitx
