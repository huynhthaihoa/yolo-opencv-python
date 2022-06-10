# YOLO object detection
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

img = cv.imread('demo.jpg')

# Load names of classes and get random colors
classLabels = open('coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classLabels), 3), dtype='uint8')

# Give the configuration and weight files for the model and load the network.
net = cv.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True, crop=False)

classes, scores, boxes = model.detect(img, 0.5, 0.4)

for (classId, score, box) in zip(classes, scores, boxes):
    color = [int(c) for c in colors[int(classId)]]
    label = "{}: {:.2f}%".format(classLabels[int(classId)], score * 100)   
    cv.rectangle(img, box, color, 4)
    cv.putText(img, label, (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 2, color, 2)

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.figure()
plt.imshow(img)
plt.show()