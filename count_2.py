from pyexpat import model

from ultralytics import YOLO
from ultralytics import solutions
import numpy as np


import cv2
import matplotlib.pyplot as plt
model = YOLO("best.pt")

def countredwhite(video_path, output_video_path, model_path, classes_to_count):
    pac= cv2.VideoCapture(video_path)
    assert pac.isOpened(), "error"
    w, h, fps = (int(pac.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter("traffic_tracked_line.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    region_points = [(700, 300),(1107,300),(1107, 800),(700, 800)]
    counter = solutions.ObjectCounter(show=True, region=region_points, model=model_path, classes=classes_to_count)
    
    LOWER_RED1 = np.array([0, 70, 70])
    UPPER_RED1 = np.array([10, 255, 255])
    LOWER_RED2 = np.array([165, 70, 70])
    UPPER_RED2 = np.array([180, 255, 255])

    # 4. Define COLOR 2: WHITE Ranges
    LOWER_WHITE = np.array([0, 0, 180])
    UPPER_WHITE = np.array([179, 40, 255])

    COLOR_THRESHOLD_red = 0.25
    COLOR_THRESHOLD_white = 0.65

    count_red = set()
    count_white=set()
    counter.boxes=[]
    ids={}



    while pac.isOpened():
        success, im0 = pac.read()
        if not success:
            print("Video frame is empty or processing is complete.")
            break

        tracks = model.track(source=im0, persist=True, tracker="bytetrack.yaml", classes=classes_to_count)
        counter.boxes=[]
        counter.clss=[]
        track_ids=[]

        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu().numpy()
            clss = tracks[0].boxes.cls.cpu().numpy()
            track_ids = tracks[0].boxes.id.cpu().numpy().astype(int)

            #for i in track_ids:
                #if i in ids:
                    #continue
                #else:
            for box, cls, track_id in zip(boxes, clss, track_ids):
                if track_id in ids:
                    continue
                x1, y1, x2, y2 = map(int, box)

                                # Crop the bounding box area
                crop = im0[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Convert crop to HSV color space
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                total_pixels = crop.shape[0] * crop.shape[1]

                mask_r1 = cv2.inRange(hsv_crop, LOWER_RED1, UPPER_RED1)
                mask_r2 = cv2.inRange(hsv_crop, LOWER_RED2, UPPER_RED2)
                red_mask = cv2.bitwise_or(mask_r1, mask_r2)
                red_ratio = np.sum(red_mask > 0) / total_pixels


                white_mask = cv2.inRange(hsv_crop, LOWER_WHITE, UPPER_WHITE)
                white_ratio = np.sum(white_mask > 0) / total_pixels

                if red_ratio > COLOR_THRESHOLD_red and track_id not in count_red:
                    ids[track_id] = cls
                    count_red.add(track_id)
                elif white_ratio > COLOR_THRESHOLD_white and track_id not in count_white:
                    ids[track_id] = cls
                    count_white.add(track_id)


        # 7. Execute the processing step on the filtered boxes
        results=counter(im0)
        processed_img = results.plot_im
        
        cv2.putText(processed_img, f"Red Cars: {len(count_red)}", (1305, 861), cv2.FONT_HERSHEY_TRIPLEX, 1, (26, 26, 188), 2)
        cv2.putText(processed_img, f"White Cars: {len(count_white)}", (1305, 957), cv2.FONT_HERSHEY_TRIPLEX, 1, (215, 225, 224), 2)

        video_writer.write(processed_img)

    pac .release()
    video_writer.release()
    cv2.destroyAllWindows()

countredwhite("traffic.mp4", "traffic_tracked_line.mp4", model, [3])
