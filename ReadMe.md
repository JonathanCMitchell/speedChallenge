Given: data/drive.mp4
8616 frames in data/IMG
each frame is 640(w) x 840(h) x 3 (RGB)

created drivinglog.csv with ['image_path', 'time', 'speed'] columns

Given ground_truth data in drive.json with [time, speed] wrapped in an array

Two methods:
1) Nvidia Model: PilotNet based implementation that compares the differences between both images and sends that through a network and performs regression based on the image differences
2) DeepVO: AlexNet like implementation that performs parallel convolutions on two images and them merges them later in the pipeline to extract special features between them

* I grabbed the model from this paper: https://arxiv.org/pdf/1611.06069.pdf

It was unclear exactly how it was being implemented so I reached out to the author and he sent me detailed notes on his network model. This is because when AlexNet was being implemented they used two vectors of length 2048 as their fully connected layers, and then multi-merged (two by two merge) them to form a 4096 dense layer. However in this implementation we use 4096 vectors as our fully connected layers and then merge them (one way merge (4096 merge 4096) to form a 8192

You can drag the train_vo.prototxt to this link: http://ethereon.github.io/netscope/#/editor
to see the network model and all its intricacies

3) DeepFlow: Large displacement optical flow with deep matching [link](http://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Weinzaepfel_DeepFlow_Large_Displacement_2013_ICCV_paper.pdf)
* I considered using DeepFlow but I found out about it literally the day before the project was due

4) Car Speed estimation using Visual Odometry

Extra:
TODO: [Keras Model Visualization] (https://keras.io/visualization/)

Step1: Run VideoToDatasetAcquisition in an ipython notebook. Just shift click your way through it. This will create a driving.csv file and an IMG folder with all the images. I did this so we can work with image paths instead of the actual images, and to make life easier

Step2: Run NvidiaModel to build the Nvidia model
* At the end of the file you will create a predict folder that holds the images with predictions overlayed on top of them. Then if you shift click through it you will create a video and then you can watch that video in the ipython notebook.



### TO TEST
# Step 1: Run ./setupstuff.sh 
* This will create the necessary folders (driving_test.csv, test_IMG, test_predict)
`./setupstuff.sh`

# Step 2: python test.py
* Go into test.py and specify the paths to video file, and the paths to the data (json) file
* This will log out the MSE
`python test.py`

# step 3: makeVideo.py
`python makeVideo.py`
* This will create a video to see how well the prediction works with the test set
* Requires moviepy


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