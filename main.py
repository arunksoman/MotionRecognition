from flask import Flask, render_template, Response, url_for
from camera import VideoCamera
from motion import MotionDetection
import cv2
import datetime
from utilities import KeyClipWriter

app = Flask(__name__)
camera = VideoCamera()

@app.route('/')
def index():
    return render_template('video_feed.html')

def gen(camera):
    kcw = KeyClipWriter(bufSize=128)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    consecFrames = 0
    firstFrame = None
    while True:
        frame = camera.get_frameAs_frame()
        updateConsecFrame = True
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if firstFrame is None:
            firstFrame = gray
            continue
        frame, motionStatus = MotionDetection(frame,firstFrame)
        if motionStatus:
            consecFrames = 0
            if not kcw.recording:
                timestamp = datetime.datetime.now()
                p = "{}-{}.avi".format("output",timestamp.strftime("%Y%m%d-%H%M%S"))
                kcw.start(p, fourcc, 20)
        if updateConsecFrame:
            consecFrames += 1
        # update the key frame clip buffer
        kcw.update(frame)
        # if we are recording and reached a threshold on consecutive
        # number of frames with no action, stop recording the clip
        if kcw.recording and consecFrames == 64:
            kcw.finish()
        frame = camera.get_frame(frame)
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # app.run(host='0.0.0.0', debug=True)
    app.run(host= '0.0.0.0', debug=True)