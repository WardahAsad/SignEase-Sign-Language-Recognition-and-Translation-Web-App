from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime
from .models import UserComment
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier




# Global Variables
detector = HandDetector(maxHands=2)
classifier = Classifier("D:/Model/keras_model_Alphabet.h5")
classifier_word = Classifier("D:/Model/CNN_keras_7_Class_model.h5", "D:/Model/labels_word_7.txt")
classifier_numbers = Classifier("D:/jangoproject/signWeb/APP1/models/keras_model_numbers.h5")

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
labels_word = ["Alert", "Bad", "Careful", "Good", "Heavy", "No", "Yes", "Unknown"]
labels_numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

predicted_sentence = ""
suggested_words = {}  # Added suggested_words dictionary
suggested_words_word = {}  # Added suggested_words_word dictionary


# Views
def home(request):
    return render(request, 'home.html')


@login_required
def service(request):
    return render(request, 'service.html')


@login_required
def user_logout(request):
    logout(request)
    return redirect('signin.html')


def about(request):
    return render(request, 'about.html')


def contact(request):
    return render(request, 'contact.html')


def Login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            if user.is_superuser:
                return redirect('admin')
            else:
                response = redirect('service')
                if "rememberme" in request.POST:
                    response.set_cookie("user_id", user.id)
                    response.set_cookie("date_login", datetime.now())
                return response
        else:
            return render(request, 'login_error.html')

    return render(request, 'signin.html')


def signup(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')

        # Check if the username or email is already taken
        if User.objects.filter(username=username).exists():
            error_message = "Username is already taken. Please choose a different one."
            return render(request, "signup.html", {'error_message': error_message})

        if User.objects.filter(email=email).exists():
            error_message = "Email is already registered. Please use a different email."
            return render(request, "signup.html", {'error_message': error_message})
        else:
            # Create the user
            user = User.objects.create_user(username=username, email=email, password=password,
                                            first_name=first_name, last_name=last_name)
            user.save()
            return redirect('signin')

    return render(request, 'signup.html')


def commentform(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone_number = request.POST.get('phone_number')
        comment = request.POST.get('comment')

        UserComment.objects.create(name=name, email=email, phone_number=phone_number, comment=comment)

    return render(request, 'base.html')


def signToText(request):
    return render(request, 'signToText.html')


def signToVoice(request):
    return render(request, 'signToVoice.html')


def TextToSign(request):
    return render(request, 'TextToSign.html')


def VoiceToSign(request):
    return render(request, 'VoiceToSign.html')


def gen_frames():
    global predicted_sentence
    predicted_sentence = ""
    cap = cv2.VideoCapture(0)

    consecutive_counts = {label: 0 for label in labels}
    letter_count = 0
    suggested_added = False

    while True:
        success, img = cap.read()
        if not success:
            continue

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((300, 300, 3), np.uint8) * 255
            imgCrop = img[y - 20:y + h + 20, x - 20:x + w + 20]
            aspect_ratio = h / w

            if aspect_ratio > 1:
                k = 300 / h
                w_cal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (w_cal, 300))
                w_gap = math.ceil((300 - w_cal) / 2)
                imgWhite[:, w_gap: w_cal + w_gap] = imgResize
            else:
                k = 300 / w
                h_cal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (300, h_cal))
                h_gap = math.ceil((300 - h_cal) / 2)
                imgWhite[h_gap: h_cal + h_gap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            if 0 <= index < len(labels):
                consecutive_counts[labels[index]] += 1
                letter_count += 1
                if consecutive_counts[labels[index]] >= 12:
                    predicted_sentence += labels[index]
                    consecutive_counts[labels[index]] = 0

                    if letter_count == 4:
                        suggested_added = False
                        first_four_letters = predicted_sentence[-4:]
                        if first_four_letters in suggested_words:
                            suggested_word = suggested_words[first_four_letters]
                            if suggested_word:
                                predicted_sentence += " " + suggested_word[0]
                                suggested_added = True

                    if suggested_added:
                        letter_count = 0

            cv2.rectangle(imgOutput, (x - 20, y - 90), (x - 20 + 400, y - 90 + 60),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)

        ret, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
               b'Content-Type: text/plain\r\n\r\n' + predicted_sentence.encode() + b'\r\n')

    cap.release()


def gen_frames_word():
    offset = 20
    imgSize = 300
    global predicted_sentence
    predicted_sentence = " "
    cap = cv2.VideoCapture(0)

    consecutive_counts_word = {label: 0 for label in labels_word}
    letter_count_word = 0
    suggested_added_word = False

    while True:
        success, img = cap.read()
        if not success:
            continue

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            prediction, index_word = classifier_word.getPrediction(imgWhite, draw=False)
            confidence_threshold = 0.75  # Set a threshold for confidence
            if max(prediction) < confidence_threshold:
                index_word = len(labels_word) - 1  # "Unknown" label index

            if 0 <= index_word < len(labels_word):
                    cv2.rectangle(imgOutput, (x - 20, y - 90), (x - 20 + 400, y - 90 + 60),
                                  (0, 255, 0), cv2.FILLED)
                    cv2.putText(imgOutput, labels_word[index_word], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                    cv2.rectangle(imgOutput, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)

                    consecutive_counts_word[labels_word[index_word]] += 1
                    letter_count_word += 1
                    if consecutive_counts_word[labels_word[index_word]] >= 5:
                        predicted_sentence += labels_word[index_word]
                        consecutive_counts_word[labels_word[index_word]] = 0

                        if letter_count_word == 4:
                            suggested_added_word = False
                            first_four_letters_word = predicted_sentence[-4:]
                            if first_four_letters_word in suggested_words_word:
                                suggested_word_word = suggested_words_word[first_four_letters_word]
                                if suggested_word_word:
                                    predicted_sentence += "  " + suggested_word_word[0]
                                    suggested_added_word = True

                            if suggested_added_word:
                                letter_count_word = 0

        ret, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
               b'Content-Type: text/plain\r\n\r\n' + predicted_sentence.encode() + b'\r\n')

    cap.release()


def gen_frames_numbers():
    offset = 20
    imgSize = 300
    global predicted_sentence
    predicted_sentence = " "
    cap = cv2.VideoCapture(0)

    consecutive_counts_word = {label: 0 for label in labels_numbers}
    letter_count_word = 0
    suggested_added_word = False

    while True:
        success, img = cap.read()
        if not success:
            continue

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            prediction, index_numbers = classifier_numbers.getPrediction(imgWhite, draw=False)
            confidence_threshold = 0.75  # Set a threshold for confidence
            if max(prediction) < confidence_threshold:
                index_numbers = len(labels_numbers) - 1  # "Unknown" label index

            if 0 <= index_numbers < len(labels_numbers):
                    cv2.rectangle(imgOutput, (x - 20, y - 90), (x - 20 + 400, y - 90 + 60),
                                  (0, 255, 0), cv2.FILLED)
                    cv2.putText(imgOutput, labels_numbers[index_numbers], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                    cv2.rectangle(imgOutput, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)

                    consecutive_counts_word[labels_numbers[index_numbers]] += 1
                    letter_count_word += 1
                    if consecutive_counts_word[labels_numbers[index_numbers]] >= 5:
                        predicted_sentence += labels_numbers[index_numbers]
                        consecutive_counts_word[labels_numbers[index_numbers]] = 0

                        if letter_count_word == 4:
                            suggested_added_word = False
                            first_four_letters_word = predicted_sentence[-4:]
                            if first_four_letters_word in suggested_words_word:
                                suggested_word_word = suggested_words_word[first_four_letters_word]
                                if suggested_word_word:
                                    predicted_sentence += "  " + suggested_word_word[0]
                                    suggested_added_word = True

                            if suggested_added_word:
                                letter_count_word = 0

        ret, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
               b'Content-Type: text/plain\r\n\r\n' + predicted_sentence.encode() + b'\r\n')

    cap.release()



def index(request):
    return render(request, 'SignToText.html')


def index_word(request):
    return render(request, 'TextToSign.html')


def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


def video_feed_word(request):
    return StreamingHttpResponse(gen_frames_word(), content_type='multipart/x-mixed-replace; boundary=frame')


def fetch_predicted_sentence(request):
    global predicted_sentence
    return JsonResponse({'predicted_sentence': predicted_sentence})


@csrf_exempt
def add_space(request):
    global predicted_sentence
    predicted_sentence += " "
    return JsonResponse({'predicted_sentence': predicted_sentence})


@csrf_exempt
def delete_letter(request):
    global predicted_sentence
    predicted_sentence = predicted_sentence[:-1]
    return JsonResponse({'predicted_sentence': predicted_sentence})






def index_numbers(request):
    return render(request, 'SignToVoice.html')


def video_feed_numbers(request):
    return StreamingHttpResponse(gen_frames_numbers(), content_type='multipart/x-mixed-replace; boundary=frame')





from django.shortcuts import render
from django.http import JsonResponse
import os

class VideoManager:
    def __init__(self):
        self.base_path = os.path.join(os.path.dirname(__file__), 'static', 'videos')
        self.video_dict = {
            "be aware it is dangerous": "dangerous",
            "wow it is decent": "decent",
            "i can't speak": "dumb",
            "i am excited": "excited",
            "this situation is fearful": "fearful",
            "i am healthy": "healthy",
            "this is important": "important",
            "you are intelligent": "intelligent",
            "this is interesting": "interesting",
            "i am late": "late",
            "it is less": "less",
            "this place is noisy": "noisy"
        }

    def get_video_path(self, key):
        key = key.lower().strip()
        folder_name = self.video_dict.get(key)
        if folder_name:
            return os.path.join(self.base_path, folder_name, "1.mp4")
        return None

video_manager = VideoManager()

def fetch_video_path(request):
    user_input = request.GET.get('text', '').lower().strip()
    print(f"User input: '{user_input}'")  # Debugging line
    video_path = video_manager.get_video_path(user_input)
    if video_path:
        relative_video_path = os.path.relpath(video_path, os.path.join(os.path.dirname(__file__), 'static')).replace('\\', '/')
        print(f"Video path found: {relative_video_path}")  # Debugging line
        return JsonResponse({'video_path': f'/static/{relative_video_path}'})
    print("No matching video found")  # Debugging line
    return JsonResponse({'error': 'No matching video found'}, status=404)

def index_sign(request):
    return render(request, 'VoiceToSign.html')


