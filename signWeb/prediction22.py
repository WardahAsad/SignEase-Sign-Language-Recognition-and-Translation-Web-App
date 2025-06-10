import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8)
classifier = Classifier("D:/jangoproject/signWeb/APP1/models/CNN_keras_7_Class_model.h5", "D:/jangoproject/signWeb/APP1/models/labels_word7.txt")
offset = 20
imgSize = 300
counter = 0

labels = ["Alert", "Bad", "Careful", "Good", "Heavy", "No", "Yes", "Unknown"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Check if hand bounding box is completely inside the frame
        if x - offset >= 0 and y - offset >= 0 and x + w + offset <= img.shape[1] and y + h + offset <= img.shape[0]:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            prediction, index_word = classifier.getPrediction(imgWhite, draw=False)
            confidence_threshold = 0.8  # Set a threshold for confidence
            if max(prediction) < confidence_threshold:
                index_word = 7  # "Unknown" label index

            print(prediction, index_word)

            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index_word], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)