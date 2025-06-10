import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize HandDetector and Classifier
detector = HandDetector(maxHands=1, detectionCon=0.8)
classifier = Classifier("C:/Users/Atsham/Desktop/FYP/keras_model.h5")
classifier.labels = ["1", "2", "3", "4", "5", "6", "7", "8","9"]

# Dictionary of suggested words for each letter
suggested_words = {
    "A": ["Apple", "Ant", "Alphabet"],
    "B": ["Ball", "Book", "Banana"],
    "C": ["Cat", "Car", "Cup"],
    "D": ["Dog", "Door", "Desk"],
    "E": ["Elephant", "Egg", "Envelope"],
    "F": ["Fish", "Fan", "Flag"],
    "G": ["Giraffe", "Guitar", "Grass"],
    "H": ["House", "Hat", "Hammer"],
    "I": ["Ice cream", "Island", "Insect"],
    "J": ["Jug", "Jellyfish", "Jacket"],
    "K": ["Kite", "Key", "King"],
    "L": ["Lion", "Lemon", "Lamp"],
    "M": ["Monkey", "Moon", "Mango"],
    "N": ["Net", "Nose", "Nest"],
    "O": ["Orange", "Owl", "Oven"],
    "P": ["Penguin", "Pencil", "Piano"],
    "Q": ["Queen", "Quilt", "Quail"],
    "R": ["Rabbit", "Rocket", "Rainbow"],
    "S": ["Star", "Sun", "Snake"],
    "T": ["Tiger", "Tree", "Train"],
    "U": ["Umbrella", "Unicorn", "Uniform"],
    "V": ["Van", "Violin", "Volcano"],
    "W": ["Whale", "Watch", "Watermelon"],
    "X": ["Xylophone", "X-ray", "Xmas"],
    "Y": ["Yak", "Yo-yo", "Yacht"],
    "Z": ["Zebra", "Zipper", "Zoo"]
}

# Constants
offset = 20
imgSize = 300
predicted_sentence = ""
consecutive_counts = {label: 0 for label in classifier.labels}
letter_count = 0  # Counter to track the number of consecutive letters
suggested_added = False  # Flag to track if suggested word is added

while True:
    # Read frame from webcam
    success, img = cap.read()
    if not success:
        print("Failed to read frame from webcam.")
        break

    # Make a copy of the frame
    imgOutput = img.copy()

    # Find hands in the frame
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Process hand gesture
        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            # Resize and preprocess the hand region
            if h > w:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Get prediction from classifier
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            # Update predicted sentence and consecutive counts
            if 0 <= index < len(classifier.labels):
                consecutive_counts[classifier.labels[index]] += 1
                letter_count += 1
                if consecutive_counts[classifier.labels[index]] >= 20:
                    predicted_sentence += classifier.labels[index]
                    consecutive_counts[classifier.labels[index]] = 0

                    # Check if the suggested word should be added
                    if letter_count == 4:
                        suggested_added = False  # Reset suggested flag
                        first_four_letters = predicted_sentence[-4:]
                        if first_four_letters in suggested_words:
                            suggested_word = suggested_words[first_four_letters]
                            if suggested_word:
                                predicted_sentence += " " + suggested_word[0]
                                suggested_added = True

                    # Reset letter count if the suggested word is added
                    if suggested_added:
                        letter_count = 0

            # Draw on output image
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, classifier.labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7,
                        (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

    # Display the output image
    cv2.imshow("Image", imgOutput)

    # Wait for key press
    key = cv2.waitKey(1)

    # Check if 's' key is pressed to add a space
    if key == ord('s'):
        predicted_sentence += " "

    # Check if 'd' key is pressed to delete the last letter
    if key == ord('d'):
        predicted_sentence = predicted_sentence[:-1]

    # Check if 'q' key is pressed to exit
    if key == ord('q'):
        break

    # Print the predicted sentence
    print("Predicted Sentence:", predicted_sentence)

# Release resources
cv2.destroyAllWindows()
cap.release()
