import tkinter as tk
import cv2
import threading
import os
import numpy as np

class VideoManager:
    def __init__(self):
        self.base_path = "D:\\jangoproject\\signWeb\\APP1\\videos"
        self.video_dict = {
            "be aware it is dangerous": "dangerous",
            "wow it is decent": "decent",
            "this is dumb": "dumb",
            "i am excited": "excited",
            "this situation is fearful": "fearful",
            "I am healthy": "healthy",
            "this is important": "important",
            "you are intelligent": "intelligent",
            "this is interesting": "interesting",
            "i am late": "late",
            "it is less": "less",
            "this place is noisy": "noisy"
        }

    def get_video_path(self, key):
        folder_name = self.video_dict.get(key)
        if folder_name:
            return os.path.join(self.base_path, folder_name, "1.mp4")
        return None

class VideoPlayer:
    def __init__(self, video_manager):
        self.video_manager = video_manager
        self.frame_width = 300
        self.frame_height = 500

    def play_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame while maintaining aspect ratio
            frame = self.resize_frame(frame, self.frame_width, self.frame_height)
            cv2.imshow('Video', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def resize_frame(self, frame, width, height):
        # Get dimensions of the frame
        h, w = frame.shape[:2]
        # Calculate the scaling factor to maintain aspect ratio
        scale = min(width / w, height / h)
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        # Resize the frame
        resized_frame = cv2.resize(frame, (new_w, new_h))
        # Create a black canvas of the desired dimensions
        canvas = np.zeros((height, width, 3), dtype="uint8")
        # Center the resized frame on the canvas
        y_offset = (height - new_h) // 2
        x_offset = (width - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame
        return canvas

    def check_input(self, user_input):
        user_input = user_input.lower()
        video_path = self.video_manager.get_video_path(user_input)
        if video_path:
            threading.Thread(target=self.play_video, args=(video_path,)).start()
        else:
            print("No matching video found for the input.")

# Setting up the tkinter GUI
class VideoApp:
    def __init__(self, root, video_player):
        self.video_player = video_player

        self.label = tk.Label(root, text="Type your sentence:")
        self.label.pack()

        self.entry = tk.Entry(root)
        self.entry.pack()

        self.button = tk.Button(root, text="Submit", command=self.on_submit)
        self.button.pack()

    def on_submit(self):
        user_input = self.entry.get()
        self.video_player.check_input(user_input)

def main():
    video_manager = VideoManager()
    video_player = VideoPlayer(video_manager)

    root = tk.Tk()
    root.title("Video Trigger")
    app = VideoApp(root, video_player)
    root.mainloop()

if __name__ == "__main__":
    main()

