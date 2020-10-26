Эта страница на [русском](./README-RU.md)

# Neural Networks for forest fire detection
<p> Early detection of the source of fire, precise localization and taking timely measures to extinguish is an urgent task. Timely detection and appropriate action is critical to preventing disasters, which can save lives and property. </>
A vision-based fire detection system captures images from cameras and immediately detects a fire, making it suitable for early fire detection. As part of this project, it is planned to implement a plug-in library for fire detection and fire propagation zones. The advantage of the developed library will be the ability to connect to various video surveillance systems based on stationary cameras.
  
## Index
1. [General description of the solution](#general-description-of-the-solution)
2. [General description of the solution logic](#general-description-of-the-solution-logic)
3. [Dataset](#dataset)
4. [The processing algorithm](#the-processing-algorithm)
5. [Object detection algorithm](#object-detection-algorithm)
6. [Conclusion](#conclusion)
7. [References](#references)

## General description of the solution
The fire detection system captures video from the camera and splits it into frames for further processing. A neural network based on convolutional layers processes each frame received. The result is a bounding box that highlights the fire area in the frame with some probability.

[:arrow_up:Index](#index)

## General description of the solution logic
<p align="center">
  <img src="https://github.com/Vladislav26Laptev/Smoke_detection/blob/main/data/Scheme-eng.png">
</>

[:arrow_up:Index](#index)

## Dataset
To train the model, video recordings were collected and marked, on which there is a fire. You can access the entries and their annotations [at this link](http://yadi.sk/d/DACCsm_-FbeYmQ?w=1).
The Dataset repository contains script:
 ````
 create_dataset_od.py
 ````
for generating the object detection network training dataset.
Following files:
 ````
 dataset_label_od.py
 ````
 ````
 train_label_od.py
 ````
 ````
 val_label_od.py
 ````
are the result of the script. To train the object detection model, you also need to run the script generate_tfrecord.py to generate a training and validation file of the tfrecord type.
````
python generate_tfrecord.py --csv_input=Dataset/dataset_label_od.csv --output_path=train/train.tfrecord --image_dir=Dataset/images_od
````

[:arrow_up:Index](#index)

## The processing algorithm
The object detection network receives one frame as an input element, for this the video stream is converted into a sequence of frames. For sequential processing, 5 or more frames are selected from the sequence with equal time intervals for sequential processing. The result will be an average prediction for all frames, since it is possible to take a frame from the sequence at the moment when the fire is not visible.

[:arrow_up:Index](#index)

## Object detection algorithm
Each selected frame is sequentially processed by the object recognition model. The following models were tested to solve the set back: EfficientDet-D1, SSD MobileNet v2, Faster R-CNN ResNet50 V1, Faster R-CNN Inception ResNet V2. 
The test result is shown in the table:

 <p align="center">
  <img src="https://github.com/Vladislav26Laptev/Smoke_detection/blob/main/data/Model.png">
</p>
 
From the results of the table it can be seen that the best results in the ratio of prediction accuracy / speed of operation are shown by the EfficientDet-D1 model. The general architecture of EfficientDet [1] largely corresponds to the paradigm of one-stage detectors. It is based on the EfficientNet model, previously trained on the ImageNet dataset. To train the network, frames with marked markings in the form of boxes with an indication of the corresponding class are required. In order to increase the efficiency of the system, namely to reduce the speed of operation and increase the accuracy of detecting fires in a forest, it is planned to consider other architectures of the EfficientDet family.

The Object Detection technology of the TensorFlow framework is used for object detection [2]. To work correctly, you need to download the [Tensorflow Object Detection repository](https://github.com/tensorflow/models/tree/master/research/object_detection), go to the project folder and start training using the following command:
````
python model_main_tf2.py --alsologtostderr --model_dir=model_od/efficientdet_d1_smoke --pipeline_config_path=model_od/efficientdet_d1/pipeline.config
````
The model was trained using the NVIDIA GeForce RTX 2080 Ti video card, and the training time was ~ 14h. The result is shown below:
<p align="center">
  <img src="https://github.com/Vladislav26Laptev/Smoke_detection/blob/main/data/Res_1.png">
</>
 
The final stage is a post-processing algorithm, the main goal of which is to combine intersecting bounding boxes using the IOU (Intersection over union) metric and combine data into clusters of 2 or more intersections to cut off false positives. As a result, we have:
 
 <p align="center">
  <img src="https://github.com/Vladislav26Laptev/Smoke_detection/blob/main/data/Res_2.png">
</>
 
[:arrow_up:Index](#index)

## Conclusion
At the moment, it is planned to develop a library based on the Python and the TensorFlow libraries for recognizing fire hazardous objects based on the researched and implemented "object detection" technologies and new approaches to improve the detection accuracy. At the moment, the accuracy of detecting fires in the test sample is 75%

[:arrow_up:Index](#index)

____
# References
1. Mingxing Tan, Ruoming Pang, Quoc V. Le, “Effi-cientDet: Scalable and Efficient Object Detection”, https://arxiv.org/abs/1911.09070
2. Object Detection | TensorFlow Hub, https://www.tensorflow.org/hub/tutorials/object_detection
