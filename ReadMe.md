Given: data/drive.mp4
8616 frames in data/IMG
each frame is 640(w) x 840(h) x 3 (RGB)

created drivinglog.csv with ['image_path', 'time', 'speed'] columns

Given ground_truth data in drive.json with [time, speed] wrapped in an array
<strong>Note: At around 2min52s when the car is on the freeway, (172 seconds in) the ground truth data declared the speed to be zero or near zero. I am unsure why this occurs in the dataset, but that is the reason why the speed is declared to be 0 at that point. Later on, around 4min 32s in (272s) the ground truth labels say the speed is ~29. The program learns both of these speeds. </strong> So I am not sure what the speedometer will have for the test_freeway data.

Approaches:
1) Nvidia Model: PilotNet based implementation that compares the differences between both images and sends that through a network and performs regression based on the image differences
2) DeepVO: AlexNet like implementation that performs parallel convolutions on two images and them merges them later in the pipeline to extract special features between them

* I grabbed the DeepVO model from this paper: https://arxiv.org/pdf/1611.06069.pdf

* You can drag the train_vo.prototxt to this link: http://ethereon.github.io/netscope/#/editor
to see the network model and all its intricacies

3) DeepFlow: Large displacement optical flow with deep matching [link](http://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Weinzaepfel_DeepFlow_Large_Displacement_2013_ICCV_paper.pdf)
* I considered using DeepFlow but I found out about it literally the day before the project was due



### To run the jupyter notebooks (how I did it)
Step1: Run VideoToDatasetAcquisition.ipynb in an ipython notebook. Just shift click your way through it. This will create a driving.csv file and an IMG folder with all the images. I did this so we can work with image paths instead of the actual images, and to make life easier

Step2: Run NvidiaModel-OpticalFlowDense-Method2.ipynb to build the Nvidia model
* At the end of the file you will create a predict folder that holds the images with predictions overlayed on top of them. Then if you shift click through it you will create a video and then you can watch that video in the ipython notebook.



### TO TEST model on new dataset
# Step 1: Run ./setupstuff.sh 
* This will create the necessary folders (driving_test.csv, test_IMG, test_predict). Note you will create a test_predict folder which will be used later if you decide to create a video from the test data
`./setupstuff.sh`

# Step 2: python test.py
* Go into test.py and specify the paths to video file, and the paths to the data (json) file. 
* This will log out the MSE
`python test.py`

# step 3: makeVideo.py
`python makeVideo.py`
* This will create a video to see how well the prediction works with the test set
* Requires moviepy

## Optical Flow 8 epoch trained model (with model-RGBM5 weights), no overlay

## Sparse Optical Flow Overlay (I had way too much fun with this) (This was on a worse performing network model)
This is using the Lucas-Kanade method for sparse optical flow analysis. The optical flow patterns are not inserted into the neural network for this exampe. 
<a href="http://www.youtube.com/embed/2XOGCPJy3Rg
" target="_blank"><img src="http://img.youtube.com/vi/2XOGCPJy3Rg/0.jpg" 
alt="Watch Video Here" width="480" height="180" border="10" /></a>



## Next steps:
Instead of feeding height x width x r,g,b into my network I will feed height x width x r,g,b x optical_flow_direction x optical_flow_magnitude into my network. TBD

## Optical Flow Analysis:
* Method 1: append images to give 3rd dimension an angular and a magnitude layer. 
In NvidiaModel-OpticalFlowDense I changed up my generator to yield (66, 220, 5) images with (Height , Width, R, G, B, Ang, Mag) Angles and Magnitudes are a result of computing the Dense Optical Flow using Farneback parameters. This did not help my MSE was still ~20 and I did not observe any special results. 

* Method 2: Convert optical flow angles and magnitude HSV to RGB and pass that into the network as (66, 220, 3) RGB values. 

* I trained this network model a few times, I started out with 12 epochs and I got an MSE on validation data ~7. I may have overfit the data though so I went back to ~8 epochs to an MSE or ~10 just to play it safe. Overfitting would be a disaster. 

### WOW
Method 2 rocks!! my MSE on validation was ~12 after 6 epochs of 20480 samples. Meaning I sent (12 * 2480 * 16) = 470k images into the network and my MSE dropped from 46.4 to 8.72. Compared to the other methods I was performing like Method 1 which dropped from 82.5996 MSE to 37.1 MSE with the same settings. Method 2 was the answer. I guess there was just too much noise when doing a simple image_1 (RGB) - image_2 (RGB). The network model held up because I converted the optical flow parameters to an RGB image. Here is a video of the results. 

### Video without sparse flow parameters
    
Implement Dense optical flow analysis, get optical flow per each pixel. as seen in [this example](http://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_lucas_kanade.html)

### Architecture Design:
![architecture design](https://github.com/JonathanCMitchell/CarND-Behavioral-Cloning-P3/blob/Master/plots/Convnet%20Architecture%20Nvidia%20Model.jpg)



#### Twitter: [@jonathancmitch](https://twitter.com/jonathancmitch)
#### Linkedin: [https://www.linkedin.com/in/jonathancmitchell](https://twitter.com/jonathancmitch)
#### Github: [github.com/jonathancmitchell](github.com/jonathancmitchell)
#### Medium: [https://medium.com/@jmitchell1991](https://medium.com/@jmitchell1991)


#### Tools used
* [Numpy](http://www.numpy.org/)
* [OpenCV3](http://pandas.pydata.org/)
* [Python](https://www.python.org/)
* [Pandas](http://pandas.pydata.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [Matplotlib](http://matplotlib.org/api/pyplot_api.html)
* [SciKit-Learn](http://scikit-learn.org/)
* [keras](http://keras.io)
