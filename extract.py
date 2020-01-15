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

n = 0
files = os.listdir(os.getcwd())
files = set(map(lambda x: x.split(".")[0], files))



shutil.rmtree(os.path.join(os.getcwd(), "Images"))
os.mkdir(os.path.join(os.getcwd(), "Images"))
os.mkdir(os.path.join(os.getcwd(), "Images", "fire"))
os.mkdir(os.path.join(os.getcwd(), "Images", "notfire"))
with open('./Images/data.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile)
    for filename in files:
    # for filename in ["house1"]:
        cur_file = glob.glob(filename + ".mp4") + glob.glob(filename + ".xml")
        if len(cur_file) == 2:
            print("Opening: {}".format(cur_file))
            video = cv2.VideoCapture(cur_file[0])
            data = bf.data(fromstring(open(cur_file[1], "r+").read()))
            
            frame_idx = 0

            while video.isOpened():
                ret, frame = video.read()
                if ret == True:
                    kp = data['opencv_storage']['frames']['_'][frame_idx]['annotations']
                    
                    if len(kp) != 0:
                        # frame = cv2.rectangle(frame, kp.split(" ")) 
                        for it, val in kp.items():
                            # print("frame_idx:", frame_idx, it, val)
                            if isinstance(val, list):
                                for idx, kplist in enumerate(val):
                                    for _, kps in kplist.items():
                                        x, y, height, width = [int(z) for z in kps.split(" ")]
                                        
                                        cv2.rectangle(frame, (x, y), (x+height, y+width), (255, 0, 0), 2)
                                        

                            else:
                                for _, kps in val.items():
                                   x, y, height, width = [int(z) for z in kps.split(" ")]
                                   cv2.rectangle(frame, (x, y), (x+height, y+width), (255, 0, 0), 2)
                        
                        
                        filename = "{}.jpg".format(n)
                        filepath = "./Images/fire/" + filename
                        filewriter.writerow(["./Images/fire/" + filename, 1])

                    else:
                        filename = "{}.jpg".format(n)
                        filepath = "./Images/notfire/" + filename

                        filewriter.writerow(["./Images/notfire/" + filename, 0])

                    # cv2.imwrite(filepath, frame)
                    n += 1
                    # cv2.imshow('frame', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    frame_idx += 1
                    # print(kp)
                else:
                    break
