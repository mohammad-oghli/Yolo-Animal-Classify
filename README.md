# YOLO Animal Recognition

This project implement **YOLO** detection and classification models in detecting and classifying image of **Single Animal** Object

The [Ultraytics Yolov11](https://docs.ultralytics.com/) pretrained models are:
* **yolo11n**: for object detection based on COCO dataset
* **yolo11n-cls**: for object classification based on Google ImageNet dataset

### Animal Classification

![animal_classify](https://i.imgur.com/sttdwJ0.jpeg)
   
  Animal Class: `'red fox'`
  
  Confidence: `68%`

The workflow for the application inference:
* `yolo11n` model is used to detect animal object by identifying its location with a bounding box in the image
* `yolo11n-cls` model is used to recognize and classify animal object in the image

There are 2 main methods in `model.py` module:
* `detection_model`: this method detects the animal object and draw object detection bounding box with classification result
* `classify_model`: this method classify the animal object in the image

Check the [classify notebook](./classify_notebook.ipynb) example for more info.

