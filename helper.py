import cv2
import numpy as np

def load_image(path):
    if type(path) != str:
        uploaded_file = path
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype="uint8")
        image = cv2.imdecode(file_bytes, -1)
    else:
        image = cv2.imread(path)  # BGR format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb
