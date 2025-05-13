import os
import sys
import cv2
import numpy as np


class FaceDetection:

    def detect_faces(src: np.ndarray, Min_Neighbour: int = 10, scale_factor: float = 1.1, min_size: int = 50):
        rectangle_thickness = 2

        ##detect faces in an image using Haar cascades.
        image = np.copy(src)

        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)

        faces = face_cascade.detectMultiScale(
            image=image,
            scaleFactor=scale_factor,
            minNeighbors=Min_Neighbour,
            minSize=(min_size, min_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        result_img = FaceDetection.draw_faces(src, faces, rectangle_thickness)

        return faces, result_img

    def draw_faces(src: np.ndarray, faces: list, thickness: int = 2):
        
        ##draw rectangles around detected faces.
        img = np.copy(src)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness)

        return img