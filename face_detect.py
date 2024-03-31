import tkinter as tk
import csv
import cv2
import os
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
from takeimage import *
from face_recog import *
import datetime
names = ['Id','Name','Date','Time']
df= pd.read_csv(f'C:\Projects\FRAS\DAMS\Databases\{std_section}\{std_section}.csv',names=names)
# Modify the trackImage function to accept std_section as an argument
def trackImage(std_section):
    # Load the face cascade classifier
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Load student details from CSV
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    # Initialize the camera
    cam = cv2.VideoCapture(0)
    count=0
    while True:
        x, y, w, h = 0, 0, 0, 0
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # Detect faces using the face cascade
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Loop through .yml files in the specified directory
            yml_directory = os.path.join(f'C:\Projects\FRAS\DAMS\Databases\{std_section}','Files')
            yml_files = [f for f in os.listdir(yml_directory) if f.endswith(".yml")]
            student_name = "Unknown"
            for yml_file in yml_files:
                recognizer = cv2.face_LBPHFaceRecognizer.create()
                recognizer.read(os.path.join(yml_directory, yml_file))
                # Recognize the face and get the predicted ID
                Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                if conf < 60:
                    count+=1
                    # Get the student's name
                    for i in range(len(df)):
                        Id1 = int(df['Id'][i])
                        if Id1 == Id:
                            student_name = df['Name'][i]
                            break  # Exit the loop once a match is found
            cv2.putText(im, student_name, (x, y + h - 10), font, 0.8, (255, 255, 255), 1)
            cv2.imshow('Face Recognition', im)
        
        if cv2.waitKey(1) & 0xFF == ord('q') or count>10:
            break
    if count>10:
        res=""
        col = ['Id','student_name','Date','Time']
        attendence = pd.DataFrame(columns=col)
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        tt = str(Id)
        attendence.loc[len(attendence)] = [Id,student_name,date,timeStamp]
        with open(f'C:\Projects\FRAS\DAMS\Databases\{std_section}\{std_section}_attendence.csv','a',newline='') as csvFile2:
            writer = csv.writer(csvFile2)
            writer.writerow([Id,student_name,date,timeStamp])
            res = "Attendence updated"
        print(res)
        
    else:
        print("attendence not updated")
    cam.release()
    cv2.destroyAllWindows()
    
trackImage(std_section)
