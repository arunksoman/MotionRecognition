import cv2
import time
import datetime
from utilities.video_utilities import VideoStream
from utilities.KeyClipWriter import KeyClipWriter
import utilities
import imutils
from queue import Queue

firstFrame = None
consecFrames=0
kcw = KeyClipWriter(bufSize=128)
print("[INFO] Camera is warming up... Please Wait...")

# Use threads to read from camera
cap = VideoStream(src=0).start()
time.sleep(3)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
while True:
    frame = cap.read()
    frame = utilities.resize(frame, width=500)
    updateConsecFrame = True
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if firstFrame is None:
        firstFrame = gray
        cv2.imshow("First Frame", firstFrame)
        continue
    cv2.imshow("Original Video", frame)
    difference = cv2.absdiff(gray, firstFrame)
    thresh = cv2.threshold(difference, 20, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(contours)
    cv2.putText(frame,time.asctime(),(0,30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        print(x)
        area = cv2.contourArea(c)
        if area>1000:
            consecFrames = 0
            cv2.putText(frame,'Motion Detected',(320,360),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,255),2,cv2.LINE_AA)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            # if we are not already recording, start recording
            if not kcw.recording:
                timestamp = datetime.datetime.now()
                p = "{}-{}.avi".format("output",timestamp.strftime("%Y%m%d-%H%M%S"))
                kcw.start(p, fourcc, 20)

    # otherwise, no action has taken place in this frame, so
    # increment the number of consecutive frames that contain
    # no action
    if updateConsecFrame:
        consecFrames += 1
    # update the key frame clip buffer
    kcw.update(frame)
    # if we are recording and reached a threshold on consecutive
    # number of frames with no action, stop recording the clip
    if kcw.recording and consecFrames == 64:
        kcw.finish()

    cv2.imshow('Contour',frame)
    cv2.imshow("Threshold", thresh)
    # cv2.imshow("Gray Frame", gray)
    cv2.imshow("Difference", difference)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        print("[Info] Turning off camera")
        break
cap.stop()
cv2.destroyAllWindows()

