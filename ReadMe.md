Given: data/drive.mp4
8616 frames in data/IMG
each frame is 640(w) x 840(h) x 3 (RGB)

created drivinglog.csv with ['image_path', 'time', 'speed'] columns

Given ground_truth data in drive.json with [time, speed] wrapped in an array
<strong>Note: At around 2min52s when the car is on the freeway, (172 seconds in) the ground truth data declared the speed to be zero or near zero. I am unsure why this occurs in the dataset, but that is the reason why the speed is declared to be 0 at that point. Later on, around 4min 32s in (272s) the ground truth labels say the speed is ~29. The program learns both of these speeds. </strong> So I am not sure what the speedometer will have for the test_freeway data.

### Method 2: 10 epoch train, (weight = `model-weights-vtest.h5`)sparse overlay video
* MSE: ~9

<a href="http://www.youtube.com/embed/1hbSINIGF3g
" target="_blank"><img src="http://img.youtube.com/vi/1hbSINIGF3g/0.jpg" 
alt="Watch Video Here" width="480" height="180" border="10" /></a>

### Method 2: 15 epoch train,(weight = `model-weights-Vtest2.h5`) sparse overlay video
* MSE: ~5.6

<a href="http://www.youtube.com/embed/wu8Q8JkDLtI
" target="_blank"><img src="http://img.youtube.com/vi/wu8Q8JkDLtI/0.jpg" 
alt="Watch Video Here" width="480" height="180" border="10" /></a>

![Mean Squared Error for v2(15 epochs)](https://github.com/JonathanCMitchell/speedChallenge/blob/master/model-vtest-2-loss.png')

### Method 2: 16 epoch train, (weight = `model-weights-RGBM3.h5`) no sparse overlay video
* MSE: ~1.24 on random 1000 sample set

<a href="http://www.youtube.com/embed/4mDY_DlIu5g
" target="_blank"><img src="http://img.youtube.com/vi/4mDY_DlIu5g/0.jpg" 
alt="Watch Video Here" width="480" height="180" border="10" /></a>


# TRAIN:
* `VideoToDatasetAcquisitionFinal` (This is what I used to write the ground truth data to a dataframe and store my images separately, this helped with testing)
* `NvidiaModel-OpticalFlowDenseM2_8Epoch.ipynb` (this is how I trained the model and demonstrated the MSE, I also processed the dataset into a video which is shown in HTML inline, notes on how I did certain things are in here)


# TEST: (also found in test_suite.zip)
* test.py
* model.py
* opticalHelpers.py
* model-weights-Vtest.h5 (trained on 10 epochs, MSE ~ 10) 
* model-weights-Vtest2.h5 (trained on 15 epochs, MSE ~ 5.6) (preloaded)
* setupstuff.sh

To test the model:
1) run `./setupstuff.sh` - this will create the necessary folders (driving_test.csv, test_IMG, test_predict)
2) create paths to your own data.json and movie.mp4 file on lines 21 and 22 inside test.py
3) `python test.py` - this will log out the MSE for a given sample size (you pick the sample size on line 14, weights should be prespecified on line 13)
4) `python makeVideo.py` - this will create a video with the prediction values overlayed on-top of each image
feel free to delete the ./data/predict folder after step 4
* Requires moviepy


model-weights-RGBM3 seems to be the best predictor, it's super easy to swap between them to see which you like the best. to swap change line 28 on test.py, and line13 on makeVideo.py to the desired weight. 


<strong>Dense Optical Flow network feeding.</strong>

### Strategies:
## Dense optical flow network feeding explanation:
* Method 1: append images to give 3rd dimension an angular and a magnitude layer. 
In NvidiaModel-OpticalFlowDense I changed up my generator to yield (66, 220, 5) images with (Height , Width, R, G, B, Ang, Mag) Angles and Magnitudes are a result of computing the Dense Optical Flow using Farneback parameters. This did not help my MSE was still ~20 and I did not observe any special results. 

* Method 2: Convert optical flow angles and magnitude HSV to RGB and pass that into the network as (66, 220, 3) RGB values. 

* I trained this network model a few times, I started out with 12 epochs and I got an MSE on validation data ~7. I may have overfit the data though so I went back to ~8 epochs to an MSE or ~10 just to play it safe. Overfitting would be a disaster. 

Method 2 rocks!! my MSE on validation was ~12 after 6 epochs of 20480 samples. Meaning I sent (12 * 2480 * 16) = 470k images into the network and my MSE dropped from 46.4 to 8.72. Compared to the other methods I was performing like Method 1 which dropped from 82.5996 MSE to 37.1 MSE with the same settings. Method 2 was the answer. I guess there was just too much noise when doing a simple image_1 (RGB) - image_2 (RGB). The network model held up because I converted the optical flow parameters to an RGB image. Here is a video of the results. 

Other approaches: 
1) Nvidia Model: PilotNet based implementation that compares the differences between both images and sends that through a network and performs regression based on the image differences
2) DeepVO: AlexNet like implementation that performs parallel convolutions on two images and them merges them later in the pipeline to extract special features between them

* I grabbed the DeepVO model from this paper: https://arxiv.org/pdf/1611.06069.pdf

* You can drag the train_vo.prototxt to this link: http://ethereon.github.io/netscope/#/editor
to see the network model and all its intricacies

3) DeepFlow: Large displacement optical flow with deep matching [link](http://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Weinzaepfel_DeepFlow_Large_Displacement_2013_ICCV_paper.pdf)
* I considered using DeepFlow

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
* moviepy
* tqdm
