import cv2 as cv
import time

# Configuration thresholds
Conf_threshold = 0.4
NMS_threshold = 0.4

# Colors for different classes
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# Load class names
class_name = []
with open('classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

# Load YOLO model
net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Open video file
cap = cv.VideoCapture('output.avi')
starting_time = time.time()
frame_counter = 0

# Process each frame
while True:
    ret, frame = cap.read()
    frame_counter += 1
    if not ret:
        break
    
    # Detect objects
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    
    # Draw bounding boxes and labels
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_name[classid], score)
        cv.rectangle(frame, box, color, 1)
        cv.putText(frame, label, (box[0], box[1] - 10), cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
    
    # Calculate FPS
    ending_time = time.time() - starting_time
    fps = frame_counter / ending_time
    cv.putText(frame, f'FPS: {fps}', (20, 50), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display frame
    cv.imshow('frame', frame)
    
    # Quit if 'q' is pressed
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
