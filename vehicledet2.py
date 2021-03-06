import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import hog
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.cross_validation import train_test_split

from scipy.ndimage.measurements import label


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True,
                                  block_norm='L1',
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True,
                       block_norm='L1',
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    if img.shape == size:
        features = img.ravel()
    else:
        features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range, density=True)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range, density=True)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range, density=True)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', patch_size=(64, 64), spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    if img.shape != patch_size:
        img = cv2.resize(img, patch_size)
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
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
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', patch_size=(64,64), spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        file_features = single_img_features(image, color_space=color_space, patch_size=patch_size,
                                            spatial_size=spatial_size,
                                            hist_bins=hist_bins, orient=orient,
                                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                                            hist_feat=hist_feat, hog_feat=hog_feat)
        features.append(file_features)
    # Return list of feature vectors
    return features
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox, or add the decision funciton output
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += box[2]

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels, heatmap):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        c = np.max(heatmap[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0]])
        conf = "{0:.3f}".format(c)
        cv2.putText(img, conf, bbox[0], cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,0), 2)
    # Return the image
    return img

def computeHeatSingle(image, box_list, visualize=False):
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,0.1)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels, heatmap)

    if visualize:
        fig = plt.figure(figsize=(20,10))
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heat, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.show()
        fig = plt.figure(figsize=(20,10))
        plt.subplot(121)
        plt.imshow(labels)
        plt.title('Output of label')
        plt.subplot(122)
        plt.imshow(draw_img)
        plt.title('Smoothed bounding box')
        plt.show()
    return draw_img


def preprocessData(color_space = 'HSV', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
          orient = 9,  # HOG orientations
          pix_per_cell = 8, # HOG pixels per cell
          cell_per_block = 2, # HOG cells per block
          hog_channel = 'ALL', # Can be 0, 1, 2, or "ALL"
          patch_size = (64, 64),
          spatial_size = (16, 16), # Spatial binning dimensions
          hist_bins = 16,    # Number of histogram bins
          spatial_feat = True, # Spatial features on or off
          hist_feat = True, # Histogram features on or off
          hog_feat = True # HOG features on or off
):
    params = {'color_space':color_space, 'orient':orient, 'pix_per_cell':pix_per_cell,
              'cell_per_block':cell_per_block, 'hog_channel':hog_channel, 'patch_size':patch_size,
              'spatial_size':spatial_size,
              'hist_bins':hist_bins}

    cars = glob.glob('vehicles/**/*.png')
    notcars = glob.glob('non-vehicles/**/*.png')


    # limit sample size for debug
    if False:
        sample_size = 500
        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]

    ### TODO: Tweak these parameters and see how the results change.

    car_features = extract_features(cars, color_space=color_space, patch_size=patch_size,
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, patch_size=patch_size,
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    # use fixed split to compare parameters
    #rand_state = 0
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    return (X_train, y_train, X_test, y_test, X_scaler, params)
    
def preprocessPCA(color_space = 'HSV', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
          orient = 9,  # HOG orientations
          pix_per_cell = 8, # HOG pixels per cell
          cell_per_block = 2, # HOG cells per block
          hog_channel = 'ALL', # Can be 0, 1, 2, or "ALL"
          patch_size = (64, 64),
          spatial_size = (16, 16), # Spatial binning dimensions
          hist_bins = 16,    # Number of histogram bins
          spatial_feat = True, # Spatial features on or off
          hist_feat = True, # Histogram features on or off
          hog_feat = True # HOG features on or off
):
    params = {'color_space':color_space, 'orient':orient, 'pix_per_cell':pix_per_cell,
              'cell_per_block':cell_per_block, 'hog_channel':hog_channel, 'patch_size':patch_size,
              'spatial_size':spatial_size,
              'hist_bins':hist_bins}

    cars = glob.glob('vehicles/**/*.png')
    notcars = glob.glob('non-vehicles/**/*.png')


    # limit sample size for debug
    if False:
        sample_size = 500
        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]

    ### TODO: Tweak these parameters and see how the results change.

    car_features = extract_features(cars, color_space=color_space, patch_size=patch_size,
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, patch_size=patch_size,
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    print(X.shape)
    pcaa = PCA() #n_components='mle',  svd_solver='full')
    pcaa.fit(X)

    # choose num components
    ncomp = np.sum(pcaa.explained_variance_ratio_ > 1e-5)
    pca = PCA(n_components = ncomp)
    X = pca.fit_transform(X)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    # use fixed split to compare parameters
    #rand_state = 0
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    params['X_scaler'] = X_scaler
    params['pca'] = pca
    return (X_train, y_train, X_test, y_test, X_scaler, params)


def train(X_train, y_train, X_test, y_test, C=0.01):
    # Use a linear SVC 
    svc = LinearSVC(C=C)
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    accuracy= round(svc.score(X_test, y_test), 4)
    print('Test Accuracy of SVC = ', accuracy)
    # Check the prediction time for a single sample
    t=time.time()
    return accuracy, svc

def gridSearch(X_train, y_train, parameters = {'C':[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}):
    #parameters = {'C':[0.00009, 0.0001, 0.00025, 0.0005, 0.00075]}
    svr = LinearSVC() #svm.SVC()
    clf = GridSearchCV(svr, parameters, verbose=1, n_jobs=4)
    t=time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    return clf.best_params_


def searchimg(filename):
    y_start_stop = [None, None] # Min and max in y to search in slide_window()

    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    draw_image = np.copy(image)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    #image = image.astype(np.float32)/255

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                        xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

    plt.imshow(window_img)


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(draw_img, img, ystart, ystop, scale, svc, X_scaler, orient,
              pix_per_cell, cell_per_block, patch_size, spatial_size,
              hist_bins, params, hog_channel='ALL', color_space='RGB'):
    
    #draw_img = np.copy(img)
    #img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    if color_space != 'RGB':
        if color_space == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)


    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # patch_size was the orginal training patch size, with 8 cells and 8 pix per cell
    window = patch_size[0]
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    #hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    #hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    #hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    if hog_channel == 'ALL':
        hog_all = []
        for channel in range(ctrans_tosearch.shape[2]):
            hog_all.append(get_hog_features(ctrans_tosearch[:,:,channel], 
                                orient, pix_per_cell, cell_per_block, 
                                vis=False, feature_vec=False))
    else:
        hog_all = get_hog_features(ctrans_tosearch[:,:,hog_channel], orient, 
                                        pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    bboxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            if hog_channel == 'ALL':
                subhog = []
                for channel in range(ctrans_tosearch.shape[2]):
#                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
#                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
#                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    subhog.append(hog_all[channel][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() )
#                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                hog_features = np.hstack(subhog)
            else:
                hog_features = hog_all[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            #subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], patch_size)
            subimg = ctrans_tosearch[ytop:ytop+window, xleft:xleft+window]
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
#            test_features = params['pca'].transform(test_features)
            test_features = X_scaler.transform(test_features)
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            confidence = svc.decision_function(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                bbox = ((xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart), confidence[0])
                bboxes.append(bbox)
    return bboxes, draw_img
    
import pickle
with open('trainedSVM64x64.p', 'rb') as file:
    params64 = pickle.load(file)


HEATMAPS = np.zeros(1)
HEATMAPSIDX = 0

def initialize(image, n=4):
    global HEATMAPS, HEATMAPSIDX
    HEATMAPS = np.zeros((n,image.shape[0],image.shape[1]), dtype=np.float)
    HEATMAPSIDX = 0

def computeHeat(image, box_list, visualize=False):
    global HEATMAPS, HEATMAPSIDX
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,box_list)
    HEATMAPS[HEATMAPSIDX,:] = heat
    HEATMAPSIDX += 1
    HEATMAPSIDX %= HEATMAPS.shape[0]

    heatavg = np.sum(HEATMAPS, axis=0) / HEATMAPS.shape[0]
    # Apply threshold to help remove false positives
    heat = apply_threshold(heatavg,0.25)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels, heat)

    if visualize:
        fig = plt.figure(figsize=(20,10))
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatavg, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.show()

        fig = plt.figure(figsize=(20,10))
        plt.subplot(121)
        plt.imshow(labels[0])
        plt.title('Output of label')
        plt.subplot(122)
        plt.imshow(draw_img)
        plt.title('Smoothed bounding box')
        plt.show()
    return draw_img

def processImage(image, visualize=False):
    color_space = params64['color_space']
    patch_size = params64['patch_size']
    spatial_size = params64['spatial_size']
    hist_bins = params64['hist_bins']
    orient = params64['orient']
    pix_per_cell = params64['pix_per_cell']
    cell_per_block = params64['cell_per_block']
    hog_channel = params64['hog_channel']
    svc = params64['svc']
    X_scalar = params64['X_scalar']

    scale = 1
    ystart=400
    ystop=500
    bb = np.copy(image)
    allbboxes = []
    bboxes, bb = find_cars(bb, image, ystart, ystop, scale, svc, X_scalar, orient, pix_per_cell, 
                           cell_per_block, patch_size, spatial_size, hist_bins, params64, hog_channel, color_space)
    
    allbboxes.extend(bboxes)
    
    scale = 2.0
    ystart=400
    ystop=650
    bboxes, bb = find_cars(bb, image, ystart, ystop, scale, svc, X_scalar, orient, pix_per_cell, 
                           cell_per_block, patch_size, spatial_size, hist_bins, params64, hog_channel, color_space)
    allbboxes.extend(bboxes)
    
    scale = 3.0
    ystart=400
    ystop=650
    bboxes, bb = find_cars(bb, image, ystart, ystop, scale, svc, X_scalar, orient, pix_per_cell, 
                           cell_per_block, patch_size, spatial_size, hist_bins, params64, hog_channel, color_space)
    allbboxes.extend(bboxes)
    
    annot_img = computeHeat(image, allbboxes, visualize=visualize)
    return annot_img

