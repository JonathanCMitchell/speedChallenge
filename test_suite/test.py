
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import json
import csv


import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# change these depending on your file names / paths
TEST_GROUND_TRUTH_JSON_PATH = './data/drive.json' # change this to the test ground truth
VIDEO_PATH = './data/drive.mp4' # change this to the test video
TEST_IMG_PATH = './test/test_IMG/'
DRIVE_TEST_CSV_PATH = './test/driving_test.csv'
TEST_PREDICT_PATH = './test/test_predict/'

# WEIGHTS = 'model-weights-F5.h5' # this one is less overfit but performs 10% worse
WEIGHTS = 'model-weights-Vtest2.h5'
EVAL_SAMPLE_SIZE = 100 # Number of samples to evaluate to compute MSE

with open(TEST_GROUND_TRUTH_JSON_PATH) as json_data:
    ground_truth = json.load(json_data)
    # json_data.close()
with open(DRIVE_TEST_CSV_PATH, 'w') as csvfile:
    fieldnames = ['image_path', 'time', 'speed']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_FRAME_COUNT, len(ground_truth))
#     cap.set(cv2.CAP_PROP_FPS, 11.7552) #11.7552


    for idx, item in enumerate(ground_truth):
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        # read in the image
        success, image = cap.read()
        if success:
            image_path = os.path.join(TEST_IMG_PATH, str(item[0]) + '.jpg')
            
            # save image to IMG folder
            cv2.imwrite(image_path, image)
            
            # write row to driving.csv
            writer.writerow({'image_path': image_path, 
                     'time':item[0],
                     'speed':item[1],
                    })
print('done writing to driving_test.csv and test_IMG folder')


### Preprocessing helpers
def preprocess_image(image):
    image_cropped = image[100:440, :-90] # -> (380, 550, 3)
    image = cv2.resize(image_cropped, (220, 66), interpolation = cv2.INTER_AREA)
    return image


def preprocess_image_valid_from_path(image_path, speed):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_image(img)
    return img, speed



from model import nvidia_model
from opticalHelpers import opticalFlowDenseDim3
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation, Dropout, Flatten, Dense, Lambda
from keras.layers import ELU
from keras.optimizers import Adam
tf.python.control_flow_ops = tf
N_img_height = 66
N_img_width = 220
N_img_channels = 3


model = nvidia_model()
model.load_weights(WEIGHTS)


### MODEL EVALUATION
samples = EVAL_SAMPLE_SIZE
errors = []
batch = []
labels = []
data = pd.read_csv(DRIVE_TEST_CSV_PATH)
indices = [np.random.randint(1, len(data) - 1) for i in range(samples)]

for idx in indices:
    row_now = data.iloc[[idx]].reset_index()
    row_prev = data.iloc[[idx - 1]].reset_index()
    row_next = data.iloc[[idx + 1]].reset_index()

    # Find the 3 respective times to determine frame order (current -> next)

    time_now = row_now['time'].values[0]
    time_prev = row_prev['time'].values[0]
    time_next = row_next['time'].values[0]

    if time_now - time_prev > 0 and 0.0000001 < time_now - time_prev < 0.58: # 0.578111 is highest diff i have seen
        # in this case row_prev is x1 and row_now is x2
        row1 = row_prev
        row2 = row_now

    elif time_next - time_now > 0 and 0.0000001 < time_next - time_now < 0.58:
        # in this case row_now is x1 and row_next is x2
        row1 = row_now
        row2 = row_next

    x1, y1 = preprocess_image_valid_from_path(row1['image_path'].values[0], row1['speed'].values[0])
    x2, y2 = preprocess_image_valid_from_path(row2['image_path'].values[0], row2['speed'].values[0])
    img_diff = opticalFlowDenseDim3(x1, x2)
    img_diff_reshaped = img_diff.reshape(1, img_diff.shape[0], img_diff.shape[1], img_diff.shape[2])
    prediction = model.predict(img_diff_reshaped)
    errors.append(np.square(abs(prediction - y2)))

MSE = np.mean(errors)
print('MSE: ', MSE)




