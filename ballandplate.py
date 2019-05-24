import cv2
import numpy as np
import sys
import time
#+sys.path.append('C:\\Users\\Pavle\\AppData\\Local\\Programs\\Python\\Python36-32\\Lib\\site-packages')
cap = cv2.VideoCapture(0)
import serial
import imutils
arduino = serial.Serial("com5", 9600)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame,width=600)
    cv2.circle(frame, (300, 225), 5, (0, 0, 255), -1)
    blurred = cv2.medianBlur(frame,5)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    ret1,img = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    #img = cv2.bilateralFilter(thresh1, 11, 30, 30)
    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    center = None
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        try:center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        except ZeroDivisionError:center = (0,0)
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    else:
        center = (-1, -1)0
    salji = str(center[0]) + "," + str(center[1]) + "."
    arduino.write(bytes(salji))
    cv2.imshow('frame', frame)
    cv2.imshow('gray', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()