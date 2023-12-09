# importing what I need for this code, I am running Python 3.11.6 and I installed Ultralytics to run this code.
import cv2
from ultralytics import YOLO
import numpy as np

# Here I am loading the path for the video I want to use
video_path = "/Users/lucasteezy/Desktop/CPSC230/final/walktest.mp4"
cap = cv2.VideoCapture(video_path)  # You can use "0" for your webcam, however I want to use my present video path, hence the variable video_path
model = YOLO("yolov8m.pt") 

# We are onlly interested in detecting people, so I use YOLO's preset dataset for detecing people
interested_classes = [0]  # The ID for 'person' in YOLO is "0"

detection_score = 0.0 #  Starts with a detection score of zero. This score will change based on whether people are detected in the video frames.
on_threshold, off_threshold = 0.7, 0.3 # These are thresholds for changing the safety status.
decay = 0.9 # This is a factor that reduces the detection score over time, assuming that if no one is detected for a while, the situation is probably safe.
# I have found that 0.9 works pretty good
safe_to_drive = True

while True: # loop is where the continuous processing of video frames happens.
    ret, frame = cap.read() # This line reads a frame from the video source. If it fails to read a frame, the loop will break
    if not ret:
        break

    # The model processes each frame to detect objects. The code specifically looks for people
    results = model(frame, device="mps")
    detections = results[0]
    bboxes = np.array(detections.boxes.xyxy.cpu(), dtype="int32")
    classes = np.array(detections.boxes.cls.cpu(), dtype="int32")

    # This checks if a person is in frame
    person_detected = any(cls in interested_classes for cls in classes)

    # This updates the detection score
    detection_score = detection_score * decay + (1 - decay) if person_detected else detection_score * decay

    # This decides whether it's safe to drive or not
    if safe_to_drive and detection_score > on_threshold:
        safe_to_drive = False
    elif not safe_to_drive and detection_score < off_threshold:
        safe_to_drive = True

    # This highlights detected people and shows status
    for cls, bbox in zip(classes, bboxes):
        if cls in interested_classes:
            x, y, x2, y2 = bbox
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, 'Person', (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    # This SHOWS whether it's safe to drive or not
    status_text = 'Safe to Drive: YES!' if safe_to_drive else 'Safe to Drive: NO!'
    cv2.putText(frame, status_text, (30, 60), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)

    # And this exit's by pressing 'ESC' key
    if cv2.waitKey(1) == 27:
        break
        
    # Some helpful output to see what's going on
    print(f'Score: {detection_score:.2f}, Safe: {safe_to_drive}')

cap.release()
cv2.destroyAllWindows()
