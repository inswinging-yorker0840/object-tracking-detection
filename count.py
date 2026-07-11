from pyexpat import model

from ultralytics import YOLO
from ultralytics import solutions


import cv2
import matplotlib.pyplot as plt
model = YOLO("best.pt")

def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

def displayRectangle(frame, bbox):
    plt.figure(figsize=(20, 10))
    frameCopy = frame.copy()
    drawRectangle(frameCopy, bbox)
    frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_RGB2BGR)
    plt.imshow(frameCopy)

vid="traffic.mp4"
pac=cv2.VideoCapture(vid)

if not pac.isOpened():
    print("Error")
    exit()

width = int(pac.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(pac.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_output_file_name = "traffic_tracked_line.mp4"
fps = pac.get(cv2.CAP_PROP_FPS)
video_out = cv2.VideoWriter(video_output_file_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
count_red = 0
count_white = 0
counter=solutions.ObjectCounter(show=True,region=[(838,338),(838,764)],model=model)

while True:
    ret, frame = pac.read()

    if not ret:
        break

    results = model(frame)
    frame=counter.count_objects(frame)
    for result in results:
        print(len(result.boxes))

        for box in result.boxes:

            class_id = int(box.cls[0])

            if class_id == 3:      # Car

                x1,y1,x2,y2 = box.xyxy[0]

                cv2.rectangle(
                    frame,
                    (int(x1),int(y1)),
                    (int(x2),int(y2)),
                    (0,0,255),
                    2
                )

    video_out.write(frame)
    