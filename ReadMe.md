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