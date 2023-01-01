import cv2
import torch
from cv2 import CascadeClassifier
from commands_wrapper import encode_text
from ImageEncoder import ImageEncoder, indexes_of_interest
import numpy as np
from utils import enable_print, print_colored_text, block_print


def c_print(text: str, color: str = 'green'):
    enable_print()
    print_colored_text(text=text, color=color)
    block_print()


class VideoStreamGuard:
    def __init__(self, input_id: int = 0):
        self.camera = cv2.VideoCapture(input_id)
        self.watch = True
        self.face_detector = CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.facial_attributes_to_look_for = None
        self.target_descriptions = None
        self.image_encoder = ImageEncoder()

    def set_monitoring_configs(self, description: str = "He's a bald man wearing eyeglasses"):
        self.target_descriptions = encode_text(description)[indexes_of_interest]

    def encode_and_compare(self, images: np.ndarray or torch.Tensor):
        results = self.image_encoder.infer(images,
                                           data_format='channels_last',
                                           return_logits=True).detach().cpu().numpy()
        enable_print()
        results = np.where(results > 0.5, 1, 0)
        percentages = [np.mean(results == self.target_descriptions) for i in range(results.shape[0])]
        return percentages

    def get_video(self):
        while self.watch:
            okay, frame = self.camera.read(cv2.IMREAD_COLOR)
            height, width = frame.shape[:2]
            padding = 0
            roi_coordinates = (height // 8, height // 8 + height * 6 // 8), (width // 8, width // 8 + width * 6 // 8)

            if self.target_descriptions is not None:
                gray = cv2.cvtColor(frame[roi_coordinates[0][0]:roi_coordinates[0][1],
                                    roi_coordinates[1][0]:roi_coordinates[1][1]],
                                    cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
                images = []
                try:
                    boarder = max(faces[0][2], faces[0][1])
                except IndexError:
                    boarder = 50

                for index, (x, y, w, h) in enumerate(faces):
                    boarder = 256
                    face = frame[y: y + boarder,
                                 x: x + boarder]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    percentages = self.encode_and_compare(np.transpose(face[np.newaxis], [0, 3, 1, 2]))
                    frame = cv2.rectangle(frame,
                                          pt1=(x,
                                               y),
                                          pt2=(x + boarder,
                                               y + boarder),
                                          color=(0,
                                                 int((1 - percentages[0]) * 255),
                                                 int(percentages[0] * 255)))
                    # frame = cv2.putText(frame, text=f'{percentages[index]}',
                    #                     org=(x + padding // 8 + max(height, width) + 50,
                    #                          y + padding // 8 + max(height, width) + 20),
                    #                     fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)

            # frame = np.clip(frame, a_min=1, a_max=254)
            if not okay:
                break

            cv2.imshow('video', frame)
            cv2.waitKey(1)


if __name__ == '__main__':
    x = VideoStreamGuard()
    x.target_descriptions = encode_text('He is a man wearing eyeglasses')[indexes_of_interest]
    x.get_video()
