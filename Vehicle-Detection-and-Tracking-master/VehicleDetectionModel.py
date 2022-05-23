from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import pickle
import cv2
import glob
import time
from ipywidgets import interact, interactive, fixed
from IPython.display import HTML
import joblib




def get_hog_features(img, orient, pix_per_cell, cell_per_block, feature_vec=True):
    """
    Extract the HOG features from the input image.
        Parameters:
            img: Input image.
            orient: Number of orientation bins.
            pix_per_cell: Size (in pixels) of a cell.
            cell_per_block: Number of cells in each block.
            vis: Visualization flag.
            feature_vec: Return the data as a feature vector.
    """
    
    features = hog(img, orientations=orient, 
                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block), 
                    transform_sqrt=True, 
                     feature_vector=feature_vec)
    return features


def bin_spatial(img, size=(16, 16)):
    """
    Compute the binned color features of the input image.
        Parameters:
            img: Input image.
            size (Default = 16 x 16): 
    """
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))




def color_hist(img, nbins=32):
    """
    Compute the color histogram features of the input image.
        Parameters:
            img: Input image.
            nbins (Default = 32): Number of histogram pins.
    """
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features




def img_features(feature_image, spatial_feat, hist_feat, hog_feat, hist_bins, orient,
                 pix_per_cell, cell_per_block, hog_channel, color_space):
    """
    Extract the features from the input image.
        Parameters:
            feature_image: Input image (RGB).
            spatial_feat: Binned color features flag.
            hist_feat: Color histogram features flag
            hog_feat: HOG features flag.
            hist_bins: Number of histogram pins.
            orient: Number of orientation bins.
            pix_per_cell: Size (in pixels) of a cell.
            cell_per_block: Number of cells in each block.
            vis: Visualization flag.
            feature_vec: Return the data as a feature vector.
            hog_channel: Number of channels per cell.
            color_space (Default = RGB): Selected color space.
    """
    file_features = []
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                feature = get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                         feature_vec=True)
                hog_features.append(feature)
                hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block,  feature_vec=True)
        file_features.append(hog_features)
    return file_features




def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Extract the features from the input images and ugment the dataset with flipped images.
        Parameters:
            imgs: Input images.
            color_space (Default = RGB): Selected color space.
            spatial_size (Default = (32, 32)): Spatial binning dimensions.
            hist_bins (Default = 32):  Number of histogram pins.
            orient (Default = 9): Number of orientation bins.
            pix_per_cell (Default = 8): Size (in pixels) of a cell.
            cell_per_block (Default = 2): Number of cells in each block.
            hog_channel (Default = 0): Number of channels per cell.
            spatial_feat: Binned color features flag.
            hist_feat: Color histogram features flag
            hog_feat: HOG features flag.
    """
    features = []
    for file_p in imgs:
        file_features = []
        image = cv2.imread(file_p)
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      
        file_features = img_features(feature_image, spatial_feat, hist_feat, hog_feat, hist_bins, orient, 
                        pix_per_cell, cell_per_block, hog_channel, color_space)
        features.append(np.concatenate(file_features))
        
        feature_image = cv2.flip(feature_image, 1) # Augment the dataset with flipped images
        file_features = img_features(feature_image, spatial_feat, hist_feat, hog_feat, hist_bins, orient, 
                        pix_per_cell, cell_per_block, hog_channel, color_space)
        features.append(np.concatenate(file_features))
    return features



def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    Generate a list of boxes with predefined parameters.
        Parameters:
            img: Input image.
            x_start_stop (Default = [None, None]): X-axis start/stop positions.
            y_start_stop (Default = [None, None]): Y-axis start/stop positions.
            xy_window (Default = 64 x 64): Window size.
            xy_overlap (Default = (0.5, 0.5)): Overlapping ratios.
    """
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            window_list.append(((startx, starty), (endx, endy)))
    return window_list




def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    Draw bounding boxes on an image.
        Parameters:
            img: Input image.
            bboxes: The bounding boxes to be drawn.
            color (Default = red): Box color.
            thick (Default = 6): Box thickness.
    """
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy




def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Extract the features of a single image.
        Parameters:
            img: Input image.
            color_space (Default = RGB): Selected color space.
            spatial_size (Default = (32, 32)): Spatial binning dimensions.
            hist_bins (Default = 32): Number of histogram pins.
            orient (Default = 9): Number of orientation bins.
            pix_per_cell (Default = 8): Size (in pixels) of a cell.
            cell_per_block (Default = 2):: Number of cells in each block.
            hog_channel (Default = 0): Number of channels per cell.
            spatial_feat (Default = True): Binned color features flag.
            hist_feat (Default = True): Color histogram features flag
            hog_feat (Default = True): HOG features flag.
        """
    img_features = []
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        elif color_space == 'LAB':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    else: feature_image = np.copy(img)      
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                     feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block,  feature_vec=True)
        img_features.append(hog_features)
    return np.concatenate(img_features)




def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, orient=8, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
    """
    Search for positive detections in the input image.
        Parameters:
            img: Input image.
            windows: A list of windows to be searched.
            scaler: The features scaler.
            color_space (Default = RGB): Selected color space.
            spatial_size (Default = (32, 32)): Spatial binning dimensions.
            hist_bins (Default = 32): Number of histogram pins.            
            orient (Default = 9): Number of orientation bins.
            pix_per_cell (Default = 8): Size (in pixels) of a cell.
            cell_per_block (Default = 2):: Number of cells in each block.
            hog_channel (Default = 0): Number of channels per cell.
            spatial_feat (Default = True): Binned color features flag.
            hist_feat (Default = True): Color histogram features flag
            hog_feat (Default = True): HOG features flag.
        """
    on_windows = []
    for window in windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = clf.predict(test_features)
        if prediction == 1:
            on_windows.append(window)
    return on_windows


def classifier_generator(color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size,
                    hist_bins, spatial_feat, hist_feat, hog_feat):
    
    t = time.time()
    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    t2 = time.time()
    
    t = time.time()
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    t2 = time.time()
    
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    # Split up data into randomized training and test sets (20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=22)
    
    
    # Use a linear SVC 
    svc = LinearSVC(loss='hinge')
    
    # Train the classifier
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    
    
    t=time.time()

    for image_f in test_images:
        image = cv2.imread(image_f)
        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 640], 
                        xy_window=(128, 128), xy_overlap=(0.85, 0.85))
        hot_windows = []
        hot_windows += (search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat))
        window_img = draw_boxes(image, hot_windows, color=(0, 0, 255), thick=6)
        
    return svc, X_scaler
        



def find_cars(img, ystart, ystop, xstart, xstop, scale, step, cspace):
    """
    Find windows with a car in a given range.
        Parameters:
            img: List of images to be displayed.
            ystart, ystop, xstart, xstop: Range to work with.
            x, y (Default = 14 x 7): Figure size.
            scale: Window scale.
            step: Wondow step.
            cspace: The selected color space.
    """
    boxes = []
    draw_img = np.zeros_like(img)   
    img_tosearch = img[ystart:ystop,xstart:xstop,:]
    if cspace != 'RGB':
        if cspace == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: ctrans_tosearch = np.copy(image)   
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))       
    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
    else: 
        ch1 = ctrans_tosearch[:,:,hog_channel]
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    window = 64
    nblocks_per_window = window // pix_per_cell - cell_per_block + 1
    cells_per_step = step  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_features = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            # Extract the image patch
            subimg = ctrans_tosearch[ytop:ytop+window, xleft:xleft+window]
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))        
            test_prediction = classifier.predict(test_features)
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)+xstart
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                boxes.append(((int(xbox_left), int(ytop_draw+ystart)),(int(xbox_left+win_draw),int(ytop_draw+win_draw+ystart))))
    return boxes


def add_heat(heatmap, bbox_list):
    """
    Filter the found windows to combine overlapping detection.
        Parameters:
            heatmap: A zero-like NumPy array with the size of the image.
            bbox_list: A list of bounding boxes.
    """
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def apply_threshold(heatmap, threshold):
    """
    Apply threshold to the heatmap to remove false positives.
        Parameters:
            heatmap: Input heatmap.
            threshold: Selected threshold.
    """
    heatmap[heatmap < threshold] = 0 
    return heatmap 

def low_filter(a, b, alpha):
    """
    Applies a simple low-pass filter.
        Parameters:
            a, b: Input coordinates and sizes.
            alpha: 
    """
    return a*alpha+(1.0-alpha)*b


def len_points(p1, p2):
    """
    Calculate the distance between two points.
        Parameters:
            p1, p2: The input points
    """
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def track_to_box(p):
    """
    Create box coordinates out of its center and span.
        Parameters:
            p: Input track.
    """
    return ((int(p[0]-p[2]),int(p[1]-p[3])),(int(p[0]+p[2]), int(p[1]+p[3])))



from scipy.ndimage.measurements import label

THRES = 3 
ALPHA = 0.75
track_list = []
THRES_LEN = 32
Y_MIN = 440

heat_p = np.zeros((720, 1280))
boxes_p = []
n_count = 0




def draw_labeled_bboxes(labels):
    """
    Generate boxes related to found cars in the frame.
        Parameters:
            labels: Input labels (Found cars).
    """
    global track_list
    track_list_l = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        #img = draw_boxes(np.copy(img), [bbox], color=(255,0,255), thick=3)
        size_x = (bbox[1][0]-bbox[0][0])/2.0 #Size of the found box
        size_y = (bbox[1][1]-bbox[0][1])/2.0
        asp_d = size_x / size_y
        size_m = (size_x + size_y)/2
        x = size_x+bbox[0][0]
        y = size_y+bbox[0][1]
        asp = (y-Y_MIN)/130.0+1.2 # Best rectangle aspect ratio for the box (coefficients from perspectieve measurements and experiments)
        if x>1050 or x<230:
            asp*=1.4
        asp = max(asp, asp_d) # for several cars chunk
        size_ya = np.sqrt(size_x*size_y/asp)
        try:
            size_xa = int(size_ya*asp)
            size_ya = int(size_ya)
        except:
            continue
        if x > (-3.049*y+1809): #If the rectangle on the road, coordinates estimated from a test image
            track_list_l.append(np.array([x, y, size_xa, size_ya]))
            if len(track_list) > 0:
                track_l = track_list_l[-1]
                dist = []
                for track in track_list:
                    dist.append(len_points(track, track_l))
                min_d = min(dist)
                if min_d < THRES_LEN:
                    ind = dist.index(min_d)
                    track_list_l[-1] = low_filter(track_list[ind], track_list_l[-1], ALPHA)
    track_list = track_list_l
    boxes = []
    for track in track_list_l:
        boxes.append(track_to_box(track))
    return boxes




def frame_processor(img, color_space):
    """
    Detects and vehicles in input image.
        Parameters:
            img: Input image.
            color_space: The color space used by the classifier.
    """
    global heat_p, boxes_p, n_count
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    boxes = []
    boxes = find_cars(img, 400, 650, 950, 1280, 2.0, 2, color_space)
    boxes += find_cars(img, 400, 500, 950, 1280, 1.5, 2, color_space)
    boxes += find_cars(img, 400, 650, 0, 330, 2.0, 2, color_space)
    boxes += find_cars(img, 400, 500, 0, 330, 1.5, 2, color_space)
    boxes += find_cars(img, 400, 460, 330, 950, 0.75, 3, color_space)
    for track in track_list:
        y_loc = track[1]+track[3]
        lane_w = (y_loc*2.841-1170.0)/3.0
        if lane_w < 96:
            lane_w = 96
        lane_h = lane_w/1.2
        lane_w = max(lane_w, track[2])
        xs = track[0]-lane_w
        xf = track[0]+lane_w
        if track[1] < Y_MIN:
            track[1] = Y_MIN
        ys = track[1]-lane_h
        yf = track[1]+lane_h
        if xs < 0: xs=0
        if xf > 1280: xf=1280
        if ys < Y_MIN - 40: ys=Y_MIN - 40
        if yf > 720: yf=720
        size_sq = lane_w / (0.015*lane_w+0.3)
        scale = size_sq / 64.0
        # Apply multi scale image windows 
        boxes+=find_cars(img, int(ys), int(yf), int(xs), int(xf), scale, 2, color_space)
        boxes+=find_cars(img, int(ys), int(yf), int(xs), int(xf), scale*1.25, 2, color_space)
        boxes+=find_cars(img, int(ys), int(yf), int(xs), int(xf), scale*1.5, 2, color_space)
        boxes+=find_cars(img, int(ys), int(yf), int(xs), int(xf), scale*1.75, 2, color_space)
    heat = add_heat(heat, boxes)
    heat_l = heat_p + heat
    heat_p = heat
    heat_l = apply_threshold(heat_l, THRES)
    heatmap = np.clip(heat_l, 0, 255)
    labels = label(heatmap)
    cars_boxes = draw_labeled_bboxes(labels)
    boxes_p = cars_boxes 
    imp = draw_boxes(np.copy(img), cars_boxes, color=(0, 0, 255), thick=6)
    n_count += 1
    return imp




from moviepy.editor import VideoFileClip
n_count = 0

def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output = cv2.cvtColor(frame_processor(image, color_space), cv2.COLOR_BGR2RGB)
    return output

color_space = 'LUV'         # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9                  # HOG orientations
pix_per_cell = 8            # HOG pixels per cell
cell_per_block = 3          # HOG cells per block
hog_channel = 0             # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)     # Spatial binning dimensions
hist_bins = 32              # Number of histogram bins
spatial_feat = True         # Spatial features flag
hist_feat = True            # Histogram features flag
hog_feat = True             # HOG features flag


classifier = joblib.load('model.sav')
X_scaler = joblib.load('scaler.gz')       
input_path=input("enter the path of the input file : ")
output_path=input("enter the path of the output file : ")


clip1 = VideoFileClip(input_path)
clip = clip1.fl_image(process_image)
clip.write_videofile(output_path, audio=False)