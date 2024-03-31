import csv
import cv2
import os
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
from takeimage import *
import shutil

# Function to trainImages


def trainImages():
    # Path for face image database
    path = paths
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(
        "C:\Projects\FRAS\DAMS\haarcascade_frontalface_default.xml"
    )
    print(path)

    # Function to get the images and label data
    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        print(imagePaths)
        faceSamples = []
        ids = []

        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert("L")
            img_numpy = np.array(PIL_img, "uint8")
            id = int(os.path.split(imagePath)[0].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for x, y, w, h in faces:
                faceSamples.append(img_numpy[y : y + h, x : x + w])
                ids.append(id)
        return faceSamples, ids

    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # write a try catch to check if the yml file exists in the directory or not
    try:
        recognizer.write(
            f"C:\Projects\FRAS\DAMS\Databases\{std_section}\Files\{std_name}.yml"
        )
    except OSError:
        print("Creation of the file failed")
    else:
        print("Successfully created the file")
    # recognizer.write(f'C:\Projects\FRAS\DAMS\Databases\{std_section}\Files\{std_name}.yml')

    shutil.rmtree(f"C:\Projects\FRAS\DAMS\Databases\{std_section}\{std_name}.{str_num}")

    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


if yml_files1:
    print("training images is not required")
else:
    trainImages()
    pass
