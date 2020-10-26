# ИИ для обнаружения лесных пожаров

[English version](#neural-networks-for-forest-fire-detection)

 <p> Раннее обнаружение источника возгорания, точная локализация и принятие своевременных мер по тушению является актуальной задачей. Своевременное обнаружение и принятие соответствующих мер имеет решающее значение для предотвращения катастроф, что влечет за собой спасения жизней и сохранение имущества людей. </>
Система обнаружения пожара на основе технического зрения захватывает изображения с камер и немедленно обнаруживает возгорание, что делает их пригодными для раннего обнаружения пожара. В этом проекте мы предлагаем метод обнаружения пожара на основе компьютерного зрения, который может работать со стационарной камерой.

## Оглавление
1. [Общее описание решения](#общее-описание-решения)
2. [Общее описание логики работы решения](#общее-описание-логики-работы-решения)
3. [Данные](#данные)
4. [Алгоритм обработки видеоряда](#алгоритм-обработки-видеоряда)
5. [Алгоритм обнаружения объекта](#алгоритм-обнаружения-объекта)
6. [Алгоритм классификации](#алгоритм-классификации)
7. [Алгоритм локализации очага](#алгоритм-локализации-очага)
8. [Итог](#итог)
9. [Источники](#источники--references)

## Общее описание решения
Система обнаружения пожара захватывает видео с камеры и разбивает его на фреймы для дальнейшей обработки. Для сохранения признаков динамики используется алгоритм вычитания фона. Обработанные кадры поступает на сверточную нейронную сеть EfficientDet-D1, обученную находить участки возгорания. Результатом является ограничивающая рамка.

[:arrow_up:Оглавление](#оглавление)

## Общее описание логики работы решения
<p align="center">
  <img src="https://github.com/Vladislav26Laptev/Smoke_detection/blob/main/data/%D0%A1%D1%85%D0%B5%D0%BC%D0%B0.png">
</>
 
[:arrow_up:Оглавление](#оглавление)
 
## Данные
Для обучения модели были собраны видеозаписи, на которых имеется возгорание. Доступ к записям и аннотациям к ним можно получить [по данной ссылке](https://yadi.sk/d/DACCsm_-FbeYmQ?w=1).
 В репозитории Dataset расположены скрипты
 ````
 create_dataset_od.py
 ````
для формирования датасета обучения сети object detection.
Файлы
 ````
 dataset_label_od.py
 ````
 ````
 train_label_od.py
 ````
 ````
 val_label_od.py
 ````
являются рзультатом работы скрипта. Для обучения модели обнаружения объекта дополнительно необходимо запустить скрипт generate_tfrecord.py для генерации тренировочного и валидационного файла типа tfrecord. 
````
python generate_tfrecord.py --csv_input=Dataset/dataset_label_od.csv --output_path=train/train.tfrecord --image_dir=Dataset/images_od
````
 
[:arrow_up:Оглавление](#оглавление)
 
## Алгоритм обработки видеоряда
Сеть обнаружения объектов в качестве входного элемента получает один кадр, для этого видеопоток преобразуется в последовательность кадров. Далее из последовательности выбираются 5 кадров с равным интервалом по времени, для последовательной обработки. Результатом будет усредненное предсказание для всех 5-и кадров, так как имеется возможность взять кадр из последовательности в момент, когда возгорания не видно.
 
[:arrow_up:Оглавление](#оглавление)
 
## Алгоритм обнаружения объекта
Каждый выбранный кадр последовательно обрабатывается моделью распознавания объектов EfficientDet-D1. Общая архитектура EfficientDet [1] в значительной степени соответствует парадигме одноступенчатых (one-stage) детекторов. За основу взята модель EfficientNet, предварительно обученная на датасете ImageNet. Для обучения сети требуются кадры с нанесенной разметкой в виде боксов с указанием соответствующего класса.
Для обнаружения объекта используется технология Object Detection фреймворка TensorFlow [2]. Для корректной работы необходимо загрузить репозиторий [Tensorflow Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection) в папку с проектом и запустить обучение используя следующую команду:
````
python model_main_tf2.py --alsologtostderr --model_dir=model_od/efficientdet_d1_smoke --pipeline_config_path=model_od/efficientdet_d1/pipeline.config
````
Обучение модели проходило с использование видеокарты NVIDIA GeForce RTX 2080 Ti, время обучения модели составило ~ 14ч. Результат работы приведен ниже:
[:arrow_up:Оглавление](#оглавление)
 
## Итог
На данный момент разработана система на базе языка Python и библиотек TensorFlow для детекния подароопасных объектов на основе технологии "object detection". Точность обнаружения на тестовой выборке составляет 75%.
 
[:arrow_up:Оглавление](#оглавление)
 
____
# Neural Networks for forest fire detection
<p> Early detection of the source of the fire, accurate localization and taking timely measures to extinguish it is an urgent task. Timely detection and appropriate action is crucial to prevent disasters, which entails saving lives and preserving people's property. </>
<p>every day we encounter a fire detection system in the form of fire and smoke sensors. They are widely used indoors and usually require the fire to burn for some time to generate a large amount of smoke and then trigger an alarm. In addition, these devices cannot be deployed outdoors on a large scale, such as in a forest.</>
There are a large number of solutions for detecting fire-hazardous objects using unmanned aerial vehicles, including using machine learning algorithms [1-2]. This project considers a system that processes data from video cameras located in a forest area.
A vision-based fire detection system captures images from cameras and immediately detects a fire, making them suitable for early fire detection. This system is cheap and easy to install. In this project, we propose a fire detection method based on computer vision that can work with a stationary camera.
A camera-based video surveillance system can monitor the specified area in real time using video processing. When the system detects a fire hazard, it sends the captured alarm image to the administrator. The administrator makes a final confirmation based on the sent alarm image.

## Index
1. [General description of the solution](#general-description-of-the-solution)
2. [General description of the solution logic](#general-description-of-the-solution-logic)
3. [Dataset](#dataset)
4. [The processing algorithm](#the-processing-algorithm)
5. [Object detection algorithm](#object-detection-algorithm)
6. [Classification algorithm](#classification-algorithm)
7. [Fire-source localization algorithm](#fire-source-localization-algorithm)
8. [Conclusion](#conclusion)
9. [References](#источники--references)

## General description of the solution
The fire detection system captures video from the camera and breaks it into frames for further processing. The background subtraction algorithm is used to preserve the dynamic features. The processed frames are sent to the EfficientDet-D1 convolutional neural network, which is trained to find fire sites. As an additional check, the previously detected section is sent to the classifier, which is a convolutional neural network based on short-term memory networks (long short-term memory, LSTM). The result is a bounding box confirmed by the classifier. The final stage of processing is the clustering algorithm that marks the epicenter of the fire.

[:arrow_up:Index](#index)

## General description of the solution logic
<p align="center">
  <img src="https://github.com/Vladislav26Laptev/Smoke_detection/blob/main/data/Scheme-eng.png">
</>

[:arrow_up:Index](#index)

## Dataset
To train the model, video recordings were collected that show a fire. You can access the entries and their annotations by [following the link](https://yadi.sk/d/DACCsm_-FbeYmQ?w=1).
 The Dataset repository contains scripts
 ````
 create_dataset_od.py
 ````
 ````
 create_dataset_cl.py
 ````
to generate the object detection network training dataset and classifier, respectively.
To train the object detection model, you also need to run the script generate_tfrecord.py to generate a training and validation file of the tfrecord type.
````
python generate_tfrecord.py --csv_input=Dataset/dataset_label_od.csv --output_path=train/train.tfrecord --image_dir=Dataset/images_od
````

[:arrow_up:Index](#index)

## The processing algorithm
Due to the dynamic nature of the fire, the shape of the smoke and flame is incorrect and constantly changing. Therefore, when using smoke as an important feature for motion detection, the usual detection methods are: continuous frame change [3], background subtraction [4], and Gaussian mixed background modeling [5]. Background subtraction is necessary to set the background correctly, because there is a large gap between day and night. The mixed Gaussian model is too complex and requires setting the historical frame, Gaussian mixture number, background refresh rate, and noise at the preprocessing stage, so this algorithm is not suitable for preprocessing, since we focus on shooting one direction for 14 seconds. The advantage of the frame difference method is its ease of implementation, low programming complexity, insensitivity to changes in the scene, such as lighting, and the ability to adapt to various dynamic environments with good stability. The disadvantage is that it is not possible to extract the entire area of the object. There is an "empty hole" inside the object, and only the border can be selected. Therefore, in this paper, an improved method of frame difference is adopted. Since the air flow and the properties of the combustion itself will cause the pixels of the flame and smoke to constantly change [6], pixel images without smoke can be removed by comparing two consecutive images.Gorenje We use an improved frame difference algorithm. First, the video stream is converted to a sequence of frames. Next, frames with a certain interval are converted from three RGB channels to one channel (grayscale transition), which saves time for calculations. In the next step, the "average frame" initialization operation is performed (1). for other images, the difference between the frame and the "average"one is used. The frame difference formula is given in (2). The output is expected to be 4 processed frames with numbers 1, 3, 5, 7 for more accurate detection. The processing result is shown below.
<p align="center">
  <img src="https://github.com/Vladislav26Laptev/Smoke_detection/blob/main/data/%D1%84%D0%BE%D1%80%D0%BC%D1%83%D0%BB%D1%8B_1.png"/>
</p>


where: 
- F_с (x,y) is the "average frame"; 
- N is the total number of frames processed;
- F_i (x,y) is the current frame of the sequence;
- F_(d_i) (x,y) is the difference between the current frame of the sequence and the average one.

<p align="center">
  <img src="https://github.com/Vladislav26Laptev/Smoke_detection/blob/main/data/%D0%A0%D0%B5%D0%B7%D1%83%D0%BB%D1%8C%D1%82%D0%B0%D1%82%20%D0%BE%D0%B1%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%BA%D0%B8.png"
       />
</p>

[:arrow_up:Index](#index)

## Object detection algorithm
After the preprocessing algorithm, each received frame is sequentially processed by the EfficientDet-D1 object recognition model. The overall architecture of EfficientDet [7] largely corresponds to the paradigm of single-stage detectors. It is based on the EfficientNet model, previously trained on the ImageNet dataset. A distinctive feature from single-stage detectors [8, 9, 10, 11], this is an additional layer with a weighted bidirectional feature pyramid (BiFPN), followed by a class and block network for generating predictions of the object class and bounding box, respectively. The box has four parameters,coordinates (x, y) for the upper-left corner and coordinates for the lower-right corner. For training the network requires personnel with printed markings in the form of boxes with indication of the corresponding class.
The object Detection technology of the TensorFlow framework is used for object detection [12]. To work correctly, you need to download the [repository](https://github.com/tensorflow/models/tree/master/research/object_detection), go to the project folder and start training using the following command:
````
python model_main_tf2.py --alsologtostderr --model_dir=model_od/efficientdet_d1_smoke --pipeline_config_path=model_od/efficientdet_d1/pipeline.config
````

[:arrow_up:Index](#index)

## Classification algorithm
As an additional check that the exposed area belongs to the class of a fire hazardous object, a classifier is being developed. The classifier is based on long short-term memory (LSTM) networks and a feature vector wrapped in a TimeDistrubed layer to analyze the features of frame dynamics. Also considered is the option of a committee of neural networks of various architectures by means of bagging (Bootstrap aggregating). This approach is considered with the aim of improving the accuracy of detection in order to eliminate false detections.

[:arrow_up:Index](#index)

## Fire-source localization algorithm
The fire-source localization algorithm at this stage of development is based on image processing using clustering algorithm. The approach is based on the assumption that the smoke in the frame spreads from the bottom up, gradually reducing the smoke density as it rises. This gives the highest intensity of smoke at the base of the fire and the lowest at the highest point.
Based on the [diagram](#general-description-of-the-solution-logic), the preliminary algorithm for finding a point is to cluster the BB area based on the color intensity. This makes it possible to highlight the smoke in the frame. Next is the most intense cluster, which will be the center of smoke (the lightest area). This cluster contains the lowest point relative to the image, which will be the source of the fire.
 <p align="center">
  <img 
       src="https://github.com/Vladislav26Laptev/Smoke_detection/blob/main/data/%D0%9A%D0%BB%D0%B0%D1%81%D1%82%D0%B5%D1%80%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F.png"
       />
</p>

[:arrow_up:Index](#index)

## Conclusion
At the moment, a plan has been developed to solve the problem, data markup has been performed, a dataset has been prepared for the object detection network, and the system architecture has been defined. While working on the project, it is planned to develop a solution based on the Python language and TensorFlow libraries.

[:arrow_up:Index](#index)

____
# Источники / References
1. Mingxing Tan, Ruoming Pang, Quoc V. Le, “Effi-cientDet: Scalable and Efficient Object Detection”, https://arxiv.org/abs/1911.09070
2. Object Detection | TensorFlow Hub, https://www.tensorflow.org/hub/tutorials/object_detection
