Given: data/drive.mp4
8616 frames in data/IMG
each frame is 640(w) x 840(h) x 3 (RGB)

created drivinglog.csv with ['image_path', 'time', 'speed'] columns

Given ground_truth data in drive.json with [time, speed] wrapped in an array

Two methods:
1) Nvidia Model: PilotNet based implementation that compares the differences between both images and sends that through a network and performs regression based on the image differences
2) DeepVO: AlexNet like implementation that performs parallel convolutions on two images and them merges them later in the pipeline to extract special features between them
