import cv2
from ultralytics import YOLO
from helper import load_image

def classify_model(image_src, loaded=False):
    classify_model = YOLO("YOLO_models/yolo11n-cls.pt")
    if not loaded:
        cls_img = load_image(image_src)
    else:
        cls_img = image_src
    classify_results = classify_model(cls_img)
    name = classify_results[0].summary()[0]['name']
    name = name.replace('_',' ')
    conf = classify_results[0].summary()[0]['confidence']
    return name, conf


def detection_model(image_src):
    detect_model = YOLO("YOLO_models/yolo11n.pt")
    detect_img = load_image(image_src)
    detect_results = detect_model(detect_img)[0]
    name, confidence = classify_model(detect_img, loaded=True)
    for box in detect_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        conf = confidence
        cls = name
        label = f'{cls} {conf:.2f}'
        # Draw rectangle and label
        cv2.rectangle(detect_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(detect_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
    return detect_img, name, confidence
        

