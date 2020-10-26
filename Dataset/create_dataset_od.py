import os
import cv2
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

frames = 15

# csv
list0 = ['ID', 'filename', 'x_min', 'y_min', 'x_max', 'y_max', 'class']
csvData = pd.DataFrame(columns=list0)
csvData_train = pd.DataFrame(columns=list0)
csvData_val = pd.DataFrame(columns=list0)

a = os.listdir('annotation/')
i = 1

list_train, list_val = train_test_split(a, test_size=0.25, random_state=42)

for j in a:
    with open('annotation/' + j) as f:
        templates = json.load(f)

    v_name = j[:-9]
    print(v_name)

    frameslist = []

    for yy in range(frames):
        b = round(len(templates['frames'])/frames*yy)
        frameslist.append(b)
        t = templates['frames'][b]
        for ii in t['figures']:
            if ii['geometryType'] == 'rectangle':
                rectangle = ii['geometry']['points']['exterior']
                x1 = rectangle[0][0]
                y1 = rectangle[0][1]
                x2 = rectangle[1][0]
                if x2 < x1:
                    z = x1
                    x1 = x2
                    x2 = z
                y2 = rectangle[1][1]
                if y2 < y1:
                    z = y1
                    y1 = y2
                    y2 = z
                log_csv = pd.DataFrame([[i, v_name + '_output_img' + '_' + str(b + 1).zfill(3) + '.png',
                                         x1, y1, x2, y2, 'smoke']], columns=list0)
                csvData = csvData.append(log_csv, ignore_index=True)
                if j in list_train:
                    csvData_train = csvData_train.append(log_csv, ignore_index=True)
                if j in list_val:
                    csvData_val = csvData_val.append(log_csv, ignore_index=True)
                i += 1

    width = templates['size']['width']
    height = templates['size']['height']

    TEST_VIDEO_PATH = os.path.join('video', v_name + '.mp4')
    cap = cv2.VideoCapture(TEST_VIDEO_PATH)

    for i in range(frames):
        cap.set(1, frameslist[i])
        b = frameslist[i]
        _, frame = cap.read()
        image_path = os.path.join('images_od/', v_name + '_output_img' + '_' + str(b + 1).zfill(3) + '.png')
        cv2.imwrite(image_path, frame)

    cap.release()
    print('Saving the video!')

csvData.to_csv('dataset_label_od.csv')
csvData_train.to_csv('train_label_od.csv')
csvData_val.to_csv('val_label_od.csv')
