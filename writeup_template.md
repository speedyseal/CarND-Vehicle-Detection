### Writeup / README

[//]: # (Image References)

[car]: ./output_images/carKITTI2980.png "Image of Car"
[not_car]: ./output_images/not_car_GTI351.png "Example of image that does not contain a car"
[hsvhog1]: ./output_images/hsvhog1.png "HOG features"
[hsvhog2]: ./output_images/hsvhog2.png "HOG features"
[hsvhog3]: ./output_images/hsvhog3.png "HOG features"
[boxdetect1]: ./output_images/boxdetect1.png "Bounding box around vehicles on I280"
[boxdetect2]: ./output_images/boxdetect2.png "Bounding box around vehicles on I280"
[heat1]: ./output_images/heat1.png "Heatmap frame 1"
[heat2]: ./output_images/heat2.png "Heatmap frame 2"
[heat3]: ./output_images/heat3.png "Heatmap frame 3"
[heat4]: ./output_images/heat4.png "Heatmap frame 4"
[label]: ./output_images/label.png "Output of label"

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines #21 through #40 of the file called `vehicledet2.py`. The function `preprocessData` on line 289 takes parameters for the feature extraction, reads in a list of filenames for `vehicle` and `non-vehicle` classes using `glob.glob` and concatenates the features using vstack. The features are scaled using StandardScalar to normalize for variance and subtract the mean bias. These are then split randomly into a test and train set using sklearn's `train_test_split`

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][car] ![alt text][not_car]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][hsvhog1]
![alt text][hsvhog2]
![alt text][hsvhog3]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. I decided to use HSV color space. Evaluating the search on full windows, I found many false positives. I felt that H and S channels were too noisy and inconsistent to have meaningful HOG signatures, whereas the outline of the vehicle is most clear in the V channel. Reducing the number of HOG features significantly reduces the total number of SVM features to train and reduces risk of overfitting.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the color histogram feature normalized to a probability density, scaled spatial features (from 64x64 to 16x16), and the HOG feature on the V channel of HSV. I used a grid search over logarithmically spaced C to find the best C parameter however for these linear SVMs, the difference in classification accuracy seems to be marginal, about 1%. The classification accuracy could vary that much depending on training set split as well.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Funciton find_cars at vehicledet2.py, line 482 performs the sliding window search. First HOG is computed for the entire region of interest in the image, which is specified by ystart, ystop for  the vertical region to search.

Then starting at one corner, windows are scanned vertically and horizontally, skipping 2 HOG cells each time. Features are constructed from the window and the HOG cells to provide to SVM for classification. Scales are determined by resizing the original image smaller to effectively increase the search area. I chose scales of 1, 2, and 3 to scan windows of 64x64, 128x128, and 192x192, which seems to encompass vehicles up close to the window. Vehicles are still identified even with window sizes smaller than the search window. As long as these windows overlap, the vehicle can still be identified.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on scales 1, 2 and 3 using HSV V-channel HOG features plus spatially binned color and normalized histograms of color in the feature vector.  Here are some example images:

The HOG is computed once to avoid recomputation of shared HOG cells. The SVM was trained on just HOG the V channel to save some computation. The SVM was trained with various settings of C using `grid_search` to find the value that yielded the best accuracy.

The following examples show the results of bounding box detection.

![alt text][boxdetect1]
![alt text][boxdetect2]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's my video result:
[![link to my video result](http://img.youtube.com/vi/fWDEjpqeEqA/0.jpg)](https://youtu.be/fWDEjpqeEqA)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the bounding boxes and SVM decision function of positive detections in each frame of the video.  From these detections I created a heatmap, using the decision function value to weight the contribution of each bounding box and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  Bounding boxes then are expanded to cover the area of each blob detected.

A parameterizable number of frames can be buffered to smooth out the detection across heatmaps. This heatmap buffer is stored in a global variable. Detections are averaged across N frames, creating a sliding moving average of detections. I used N=4.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are 4 frames and their corresponding heatmaps:

![alt text][heat1]
![alt text][heat2]
![alt text][heat3]
![alt text][heat4]


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all 4 frames, with the resulting bounding boxes drawn onto the last frame in the series:
![alt text][label]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are many false positives on patches of the image that contain trees, and road. Also when the white car is on the concrete segment of the freeway, detection drops out. It could be that the training set only has cars on asphalt, or a large fraction of the training set is backlit. SVM works great on medium sized data sets but is less effective with large datasets so it is important to review the data set more carefully.

It may be necessary to have exposure correction and normalization. To be accurate, each patch for the classification should be independently normalized, however this is an expensive operation. Adaptive equalization on each frame is also rather expensive and the search procedure is already slow as it is.

It would be worth exploring the parameter space of PCA dimensionality reduction to reduce the risk of overfitting, however a brief exploration into this yielded poor results (90% detection). More time needs to be spent on determining a set of basis vectors that retains classification accuracy.

When cars overlap in the image, the bounding box combines, detecting only 1 vehicle instead of 2. Vehicles on the other side of the road can also be detected, even through the grate in the divider. These kinds of detector behavior can be misleading to the path planning component of the system. Bounding box detection can be informed by differences between successive frames. A vehicle is unlikely to disappear in the middle of the screen but the bounding box detection may lose track for several frames at a time. Using a predictive model of vehicle motion, for example, a Kalman filter, can help bounding box tracking or disambiguate situations where bounding boxes merge or disappear.

State of the art approaches are convnet based segmentation or bounding box detection such as SSD https://arxiv.org/pdf/1512.02325.pdf or YOLO https://arxiv.org/pdf/1506.02640.pdf. These have the potential for incredible detection speedup, parallelizing the search across convnet matrix operations on a GPU.
