width = int(cp.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cp.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_output_file_name = "race_car-tracked.mp4"
fps = cp.get(cv2.CAP_PROP_FPS)
video_out = cv2.VideoWriter(video_output_file_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))