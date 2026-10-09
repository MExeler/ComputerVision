import os
import cv2
from ultralytics import solutions

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

stream_url = "rtsp://admin:Internet1!@10.88.0.111"
cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
#cap = cv2.VideoCapture("sheep.mp4")
#cap = cv2.VideoCapture(stream_url)
assert cap.isOpened(), "Error reading video file"


#region_points = [(20, 400), (1080, 400)]                                      # line counting
#region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # rectangular region
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # polygon region

# Video writer
src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 20


frame_step = 1
outputfps = fps / frame_step

width = 40
h=480
w=640


#region_points = [(w // 2, 0), (w // 2, h)]
region_points = [(w // 2 - width // 2, 0), (w // 2 + width // 2, 0), (w // 2 + width // 2, h), (w // 2 - width // 2, h)]
video_writer = cv2.VideoWriter(
    "output.avi",
    cv2.VideoWriter_fourcc(*"mp4v"),
    outputfps,
    (w, h),
)

# Initialize object counter object
counter = solutions.ObjectCounter(
    show=True,  # display the output
    region=region_points,  # pass region points
    model="yolo26n.pt",  # model="yolo26n-obb.pt" for object counting with OBB model.
    classes=[18],  # count specific classes, e.g., person and car with the COCO pretrained model.
    tracker="botsort.yaml",  # choose trackers, e.g., "bytetrack.yaml"
    show_boxes = True,
    #device=0 #should choose gpu if available
    show_in =True,
    show_out = True,
    show_labels = True,
    #imgsz=320,
    #imgsz=(h,w),
)



# Process video
frame_index = 0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break
    frame_index += 1
    if frame_index % frame_step != 0:
        continue
    im0 = cv2.resize(im0, (w, h))
    print()
    results = counter(im0)
    video_writer.write(results.plot_im)


cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows
