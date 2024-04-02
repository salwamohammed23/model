from ultralytics import YOLO
import cv2
import cvzone
import math
from streamlit_webrtc import webrtc_streamer,RTCConfiguration, WebRtcMode
import av
from twilio_tokens import get_ice_servers

model = YOLO('best.pt')

# cap = cv2.VideoCapture(0)
# cap.set(3,640) # width
# cap.set(4,360) # height

class VideoProcess():
    def recv(self, frame):
        frm = frame.to_ndarray(format='bgr24')
        results = model(frm, stream=True, verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # opencv
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img, (x1, y1), (x2, y2), (0,200,0), 3)

                # cvzone
                w, h = x2-x1, y2-y1
                cvzone.cornerRect(frm, (x1, y1, w, h))

                conf = math.ceil(box.conf[0])
                # Class Name
                cls = int(box.cls[0])

                cvzone.putTextRect(frm, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
        return av.VideoFrame.from_ndarray(frm, format='bgr24')

classNames = ["snake", "scrobian" ]

