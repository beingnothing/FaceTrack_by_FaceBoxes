%load_ext autoreload
%autoreload 2

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import numpy as np
from PIL import Image, ImageDraw
import os
import cv2
import time

from face_detector import FaceDetector

MODEL_PATH = 'model.pb'
face_detector = FaceDetector(MODEL_PATH, gpu_memory_fraction=0.25, visible_device_list='0')

# Get an image
path = '/home/gpu2/hdd/dan/WIDER/WIDER_train/images/48--Parachutist_Paratrooper/48_Parachutist_Paratrooper_Parachutist_Paratrooper_48_972.jpg'

image_array = cv2.imread(path)
image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image_array)
image

# Show detections
def draw_boxes_on_image(image, boxes, scores):

    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy, 'RGBA')
    width, height = image.size

    for b, s in zip(boxes, scores):
        ymin, xmin, ymax, xmax = b
        fill = (255, 0, 0, 45)
        outline = 'red'
        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            fill=fill, outline=outline
        )
        draw.text((xmin, ymin), text='{:.3f}'.format(s))
    return image_copy


boxes, scores = face_detector(image_array, score_threshold=0.3)
draw_boxes_on_image(Image.fromarray(image_array), boxes, scores)

# Measure speed
times = []
for _ in range(110):
    start = time.perf_counter()
    boxes, scores = face_detector(image_array, score_threshold=0.25)
    times.append(time.perf_counter() - start)
    
times = np.array(times)
times = times[10:]
print(times.mean(), times.std())