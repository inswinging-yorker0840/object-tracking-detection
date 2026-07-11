import cv2
import matplotlib.pyplot as plt


tracker = cv2.TrackerCSRT.create()

#to draw rectangle
def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

#to display rectangle
def displayRectangle(frame, bbox):
    plt.figure(figsize=(20, 10))
    frameCopy = frame.copy()
    drawRectangle(frameCopy, bbox)
    frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_RGB2BGR)
    plt.imshow(frameCopy)


video = "suburb.mp4"
cp=cv2.VideoCapture(video)
print(cp)



if not cp.isOpened():
    print("Error")
    exit()


ret, frame = cp.read()
print(ret)
print(frame)


if not ret:
    print("End of video")
    exit()
width = int(cp.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cp.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_output_file_name = "race_car-tracked.mp4"
fps = cp.get(cv2.CAP_PROP_FPS)
video_out = cv2.VideoWriter(video_output_file_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

bbox = (890, 938, 40, 95)
#displayRectangle(frame, bbox)

ok = tracker.init(frame, bbox)
video_out.write(frame)
while True:
    ok, frame = cp.read()

    if not ok:
        break

    timer = cv2.getTickCount()

    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Draw bounding box
    if ok:
        drawRectangle(frame, bbox)
    else:
        print("Tracking failure detected")


        # Write frame to video
    video_out.write(frame)
print(video_out.isOpened())

cp.release()
video_out.release()
