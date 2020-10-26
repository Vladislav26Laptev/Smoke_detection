Эта страница на [русском](./README-RU.md)

# Neural Networks for forest fire detection
<p> Early detection of the source of the fire, accurate localization and taking timely measures to extinguish it is an urgent task. Timely detection and appropriate action is crucial to prevent disasters, which entails saving lives and preserving people's property. </>
A vision-based fire detection system captures images from cameras and immediately detects a fire, making it suitable for early fire detection. In this project, we plan to implement an automated system for detecting fires and fire propagation zones based on computer vision detection methods that work with stationary cameras.
  
## Index
1. [General description of the solution](#general-description-of-the-solution)
2. [General description of the solution logic](#general-description-of-the-solution-logic)
3. [Dataset](#dataset)
4. [The processing algorithm](#the-processing-algorithm)
5. [Object detection algorithm](#object-detection-algorithm)
6. [Conclusion](#conclusion)
7. [References](#references)

## General description of the solution
The fire detection system captures video from the camera and breaks it into frames for further processing. A neural network based on convolutional layers processes each received frame. The result is a bounding box that highlights the area of fire in the frame and the point of localization of the fire with some probability.

[:arrow_up:Index](#index)

## General description of the solution logic
<p align="center">
  <img src="https://github.com/Vladislav26Laptev/Smoke_detection/blob/main/data/Scheme-eng.png">
</>

[:arrow_up:Index](#index)

## Dataset
To train the model, video recordings were collected and marked, on which there is a fire. You can access the entries and their annotations [at this link](http://yadi.sk/d/DACCsm_-FbeYmQ?w=1).
The Dataset repository contains scripts
 ````
 create_dataset_od.py
 ````
for generating the object detection network training dataset.
The files
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
The object detection network receives a single frame as an input element. To do this, the video stream is converted to a sequence of frames. Next, 5 frames are selected from the sequence with an equal time interval for sequential processing. The result will be an average prediction for all 5 frames, since it is possible to take a frame from the sequence at a moment when the fire is not visible.

[:arrow_up:Index](#index)

## Object detection algorithm
Each selected frame is sequentially processed by the object recognition model. The following models were tested to solve the set back: EfficientDet-D1, SSD MobileNet v2, Faster R-CNN ResNet50 V1, Faster R-CNN Inception ResNet V2. 
The test result is shown in the table:

 <p align="center">
  <img src="https://github.com/Vladislav26Laptev/Smoke_detection/blob/main/data/Model.png">
</p>
 
It can be seen from the results of the table that the best results in the ratio of prediction accuracy / speed of operation are shown by the EfficientDet-D1 model. The overall architecture of EfficientDet [1] largely corresponds to the paradigm of single-stage detectors. It is based on the EfficientNet model, previously trained on the ImageNet dataset. For training the network requires personnel with printed markings in the form of boxes with indication of the corresponding class.
The object Detection technology of the TensorFlow framework is used for object detection [2]. To work correctly, you need to download the [Tensorflow Object Detection repository](https://github.com/tensorflow/models/tree/master/research/object_detection), go to the project folder and start training using the following command:
````
python model_main_tf2.py --alsologtostderr --model_dir=model_od/efficientdet_d1_smoke --pipeline_config_path=model_od/efficientdet_d1/pipeline.config
````
The model was trained using the NVIDIA GeForce RTX 2080 Ti video card, and the training time was ~ 14h. The result is shown below:
<p align="center">
  <img src="https://github.com/Vladislav26Laptev/Smoke_detection/blob/main/data/Res_1.png">
</>
 
The final stage is the post-processing algorithm, the main task of which is to connect intersecting bounding boxes using the IOU (Intersection over union) metric and combine data into clusters of 2 or more intersections to cut off false positives. As a result, we have:
 
 <p align="center">
  <img src="https://github.com/Vladislav26Laptev/Smoke_detection/blob/main/data/Res_2.png">
</>
 
[:arrow_up:Index](#index)

## Conclusion
Currently, a system based on the Python language and TensorFlow libraries has been developed for recognizing fire-hazardous objects based on the "object detection"technology. The accuracy of fire detection in the test sample is 75%.

[:arrow_up:Index](#index)

____
# References
1. Mingxing Tan, Ruoming Pang, Quoc V. Le, “Effi-cientDet: Scalable and Efficient Object Detection”, https://arxiv.org/abs/1911.09070
2. Object Detection | TensorFlow Hub, https://www.tensorflow.org/hub/tutorials/object_detection
