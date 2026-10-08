import cv2
from ultralytics import solutions

#stream_url = "rtsp://admin:Internet1!@10.88.0.111"
#cap = cv2.VideoCapture(stream_url)
cap = cv2.VideoCapture("sheep.mp4")
assert cap.isOpened(), "Error reading video file"


#region_points = [(20, 400), (1080, 400)]                                      # line counting
#region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # rectangular region
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # polygon region

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

region_points = [(w // 2, 0),(w // 2, h)]
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize object counter object
counter = solutions.ObjectCounter(
    show=True,  # display the output
    region=region_points,  # pass region points
    model="yolo26n.pt",  # model="yolo26n-obb.pt" for object counting with OBB model.
    classes=[18],  # count specific classes, e.g., person and car with the COCO pretrained model.
    tracker="botsort.yaml",  # choose trackers, e.g., "bytetrack.yaml"
    show_boxes = True,
    show_in =True,
   # device=0
)




# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = counter(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows