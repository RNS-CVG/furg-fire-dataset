import glob
import os
import cv2
import json
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import fromstring
from xmljson import badgerfish as bf
import csv
import shutil
import numpy as np
from PIL import Image
import random
import math

n = 0
positives = 0
files = os.listdir(os.getcwd())
files = set(map(lambda x: x.split(".")[0], files))

def removeAndCreate(folderName):
    if os.path.exists(folderName):
        shutil.rmtree(folderName)
    os.mkdir(folderName)

def centerAndNormalize(x, y, h, w, Ih, Iw):
    x_center = (x + (h / 2)) / Iw
    y_center = (y + (w / 2)) / Ih
    h = (h / Iw)
    w = (w / Ih)

    print(x_center, y_center, h, w, Ih, Iw)


    assert x_center  > 0 and x_center < 1
    assert y_center  > 0 and y_center < 1
    assert h >= 0 and h <= 1
    assert w >= 0 and w <= 1

    
    return x_center, y_center, h, w 

removeAndCreate("coco")
removeAndCreate("coco/images")
removeAndCreate("coco/labels")
with open("./coco/train.txt", "x") as train, open("./coco/test.txt", "x") as test:
    print("Train, test folders created")
with open("./coco/data.names", "x") as data:
    data.write("fire")

for filename in files:
# for filename in ["Car1"]:
    cur_file = glob.glob(filename + ".mp4") + glob.glob(filename + ".xml")
    if len(cur_file) == 2:
        print("Opening: {}".format(cur_file))
        video = cv2.VideoCapture(cur_file[0])
        data = bf.data(fromstring(open(cur_file[1], "r+").read()))
        
        frame_idx = 0

        while video.isOpened():
            ret, frame = video.read()
            train_frames, test_frames = 0, 1
            if ret == True:
                kp = data['opencv_storage']['frames']['_'][frame_idx]['annotations']
                datalist = []
                if len(kp) != 0:
                    # frame = cv2.rectangle(frame, kp.split(" ")) 
                    for it, val in kp.items():
                        # print("frame_idx:", frame_idx, it, val)

                        if isinstance(val, list):
                            for idx, kplist in enumerate(val):
                                for _, kps in kplist.items():
                                    x, y, height, width = [int(z) for z in kps.split(" ")]
                                    x_n, y_n, w_n, h_n = centerAndNormalize(x, y, height, width, frame.shape[0], frame.shape[1])
                                    datalist.append(("0 {} {} {} {}".format(x_n, y_n, w_n, h_n)))
                                
                                    # cv2.circle(frame, (int(x_n), int(y_n)), 5, (0, 255, 0), 2)
                                    # cv2.rectangle(frame, (x, y), (x+height, y+width), (255, 0, 0), 2)
                                    
                        else:
                            for _, kps in val.items():
                                x, y, height, width = [int(z) for z in kps.split(" ")]
                                x_n, y_n, w_n, h_n = centerAndNormalize(x, y, height, width, frame.shape[0], frame.shape[1])
                                datalist.append(("0 {} {} {} {}".format(x_n, y_n, w_n, h_n)))
                                # cv2.circle(frame, (int(x_n), int(y_n)), 5, (0, 255, 0), 2)
                                # cv2.rectangle(frame, (x, y), (x+height, y+width), (255, 0, 0), 2)
                
                print("%: {}".format(n / 26223 * 100), positives)
                if positives % 180 == 0 and len(datalist) != 0:
                    filename = "{}.jpg".format(n)
                    filepath = "./coco/images/" + filename
                    cv2.imwrite(filepath, frame)

                    if len(datalist) != 0:
                        
                        with open("./coco/labels/{}.txt".format(n), "x") as datafile:
                            datafile.write("\n".join(datalist))
                        
                        with open("./coco/train.txt", "a") as train, open("./coco/test.txt", "a") as test:
                            if random.choice([x for x in range(1, 10)]) > 2:
                                train.write(filepath + "\n")
                            else:
                                test.write(filepath + "\n")

                
                if len(datalist) != 0: positives += 1
                n += 1
                
                # cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_idx += 1
                # print(kp)
            else:
                break
