import cv2, time
import random
import os
import numpy as np
from datetime import datetime as dt
from tkinter import *
from tkinter import messagebox
import time
import pandas as pd

# window = Tk()
#
# messagebox.showinfo("Ready", "Get ready in 5 second")
#
# window.after(3000, lambda: window.destroy())
#
# window.mainloop()

df = pd.DataFrame(columns=['Start', 'End'])

first_frame = None

video = cv2.VideoCapture(0)

# this variable is used for the alarm
continuous_present = False

time = []


while True:
    check, frame = video.read()
    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # use GussianBlur to smooth the image
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # store thr first frame for comparison
    if first_frame is None:
        first_frame = gray
        continue

    # delta frame is the difference between the background (first-frame), and the subsequent frame
    delta_frame = cv2.absdiff(first_frame, gray)

    # set threshold for the delta_frame (I set it to 100 to make it less sensitive)
    ret, thresh_frame = cv2.threshold(delta_frame, 100, 255, cv2.THRESH_BINARY)

    # make the white spot go away
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=10)

    # find the contour of all the detected object
    (cnts, a) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if no object present, sent it to False
    if len(cnts) == 0 and continuous_present == True:
        continuous_present = False
        # create timestamp when object leave
        time.append(dt.now())


    for contour in cnts:
        # print(cv2.contourArea(contour))
        if cv2.contourArea(contour) < 20000:
            continue
        else:
            status = 1
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.drawContours(frame, cnts, -1, (0, 255, 0), 3)
            if status == 1 and continuous_present == False:
                # create timestampe when object enter
                time.append(dt.now())
                print(status, continuous_present)
                os.system('afplay /System/Library/Sounds/Sosumi.aiff')
                continuous_present = True
            # print("someone in the vicinity", str(random.randint(0, 100000)))
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.imshow("Capturing", gray)
    cv2.imshow("delta_frame", delta_frame)
    cv2.imshow("Thresh_delta", thresh_frame)
    cv2.imshow("Color Frame", frame)
    cv2.moveWindow("Color Frame", 20, 20)

    key = cv2.waitKey(1)
    # print(gray)

    if key == ord('q'):
        # if person quit the program without leaving the frame
        if continuous_present == True:
            time.append(dt.now())
        break

print(time)

for i in range(0, len(time), 2):
    df = df.append({"Start": time[i], "End": time[i+1]}, ignore_index=True)


df.to_csv("Times.csv")
print(df)
video.release()
cv2.destroyAllWindows
