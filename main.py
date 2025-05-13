import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QMessageBox
from PyQt5.uic import loadUiType
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import logging

from face_detection import FaceDetection
from face_recognition import FaceRecognition


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ui, _ = loadUiType("faceDetRecUI.ui")

class MainApp(QtWidgets.QMainWindow, ui):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)

        self.image = None
        self.face_recognition = FaceRecognition()
        self.test_images = []
        self.test_image_paths = []
        self.current_test_image_index = -1
        
        self.upload_button.clicked.connect(self.upload)
        self.applyDetection_button.clicked.connect(self.apply_face_detection)
        self.uploadImage1_button.clicked.connect(self.upload_training_data)
        self.uploadImage2_button.clicked.connect(self.make_eigen_faces)
        self.pushButton.clicked.connect(self.load_single_test_image)
        self.applyMethods_button.clicked.connect(self.match_faces)
        
        self.statusBar().showMessage("Ready")

    def upload(self): #single image uploading for face detection
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", 
            "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.webp *.pgm);;All Files (*)", 
            options=options
        )

        if file_path:
            self.q_image, self.image = self.process_and_store_image(file_path)  
            self.inputDetection.setPixmap(QPixmap.fromImage(self.q_image))
            self.outputDetection.setPixmap(QPixmap())
            self.inputDetection.setScaledContents(True)
            self.outputDetection.setScaledContents(True)
            
            self.statusBar().showMessage(f"Loaded image: {os.path.basename(file_path)}")
            logger.info(f"Uploaded image from {file_path}")

    def process_and_store_image(self, file_path):
        original_image = Image.open(file_path).convert("RGB")
        img_array = np.array(original_image)
    
        # Convert PIL image to QImage
        height, width, channels = img_array.shape
        bytes_per_line = channels * width
        q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        return q_image, img_array
    
    def apply_face_detection(self): #applying facce detection to currently uploaded image
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please upload an image first.")
            return
        
        Min_Neighbour = int(self.tickness_lineEdit.text())
        scale_factor = float(self.scaleFactor_lineEdit.text())
        window_size = int(self.windowSize_lineEdit.text())
        
        start = time.time()
        faces, detected_faces = FaceDetection.detect_faces(self.image, Min_Neighbour, scale_factor, window_size)
        end = time.time()
        
        self.display_result_on_label(self.outputDetection, detected_faces)
        elapsed_ms = (end - start) * 1000  # Convert to milliseconds
        self.detectionTime_label.setText(f"{elapsed_ms:.2f} ms")
        
        self.statusBar().showMessage(f"Detected {len(faces)} faces in {elapsed_ms:.2f} ms")
        logger.info(f"Face detection completed. Found {len(faces)} faces.")


    def display_result_on_label(self, label: QLabel, image: np.ndarray):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        if len(image.shape) == 2:  # Grayscale
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        elif len(image.shape) == 3 and image.shape[2] == 3:  # Color
            height, width, channels = image.shape
            bytes_per_line = channels * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

    def upload_training_data(self): #uploading training data for face recognition
        folder_path = QFileDialog.getExistingDirectory(self, "Select Training Data Folder")
        success, message, images_data = self.face_recognition.load_training_data(folder_path)
        
        if success:
            self.statusBar().showMessage(message)
            self.uploadImage2_button.setEnabled(True) #enabling eigenfaces button
            self.face_recognition.visualize_training_data(images_data) #visualizing training data



    def make_eigen_faces(self): #generating eigenfaces using PCA
        success, message = self.face_recognition.compute_eigenfaces()
        if success:
            self.statusBar().showMessage(message) 
            # Enable test data loading after eigenfaces are computed
            self.pushButton.setEnabled(True) #enabling loading testing data button


    def load_single_test_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Test Image", "", 
            "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.webp *.pgm);;All Files (*)", 
            options=options
        )
        
        if file_path:
            img = cv2.imread(file_path)
            if img is not None:
                self.test_image = img
                self.test_image_path = file_path
                
                # converting to RGB for Qt display
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                height, width, channels = img_rgb.shape
                bytes_per_line = channels * width
                q_image = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                
                # displaying in both input labels
                self.inputDetection.setPixmap(QPixmap.fromImage(q_image))
                self.inputDetection.setScaledContents(True)
                self.inputRecognition.setPixmap(QPixmap.fromImage(q_image))
                self.inputRecognition.setScaledContents(True)
                
                # clearing output labels
                self.outputDetection.setPixmap(QPixmap())
                self.outputRecognition.setPixmap(QPixmap())
                
                # storing current image for processing
                self.image = img_rgb
                
                # enabling processing button
                self.applyMethods_button.setEnabled(True)
                
                # showing image info
                file_name = os.path.basename(file_path)
                self.statusBar().showMessage(f"Loaded test image: {file_name}")
            else:
                self.statusBar().showMessage("Failed to load image")

    def match_faces(self):  #performing face recognition to test image
        # converting RGB to BGR for OpenCV processing if needed
        if len(self.image.shape) == 3 and self.image.shape[2] == 3:
            img_for_recognition = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        else:
            img_for_recognition = self.image.copy()

        #performing face recognition
        recognized_image, results, best_match_image = self.face_recognition.recognize_face(img_for_recognition)

        if recognized_image is not None:
            # converting recognized image back to RGB for display
            if len(recognized_image.shape) == 3 and recognized_image.shape[2] == 3:
                recognized_image_rgb = cv2.cvtColor(recognized_image, cv2.COLOR_BGR2RGB)
            else:
                recognized_image_rgb = recognized_image

            # prepare training match image
            if best_match_image is not None:
                if len(best_match_image.shape) == 2:
                    best_match_image_rgb = cv2.cvtColor(best_match_image, cv2.COLOR_GRAY2RGB)
                else:
                    best_match_image_rgb = cv2.cvtColor(best_match_image, cv2.COLOR_BGR2RGB)
            else:
                best_match_image_rgb = None

            # showing test image with rectangles on left label
            self.display_result_on_label(self.inputRecognition, recognized_image_rgb)

            # showing training match image on right label 
            if best_match_image_rgb is not None:
                self.display_result_on_label(self.outputRecognition, best_match_image_rgb)

            # showing recognition results
            result_text = ""
            if results:
                for face_idx, (label, confidence) in enumerate(results):
                    subject = self.face_recognition.subject_names.get(label, f"Unknown (Label {label})")
                    result_text += f"Face {face_idx+1}: {subject}, Confidence: {confidence:.2f}\n"

                first_subject = self.face_recognition.subject_names.get(results[0][0], f"Unknown (Label {results[0][0]})")
                self.statusBar().showMessage(f"Recognized as {first_subject} (Confidence: {results[0][1]:.2f})")
            else:
                self.statusBar().showMessage("No faces recognized in the image")

            if result_text:
                QMessageBox.information(self, "Recognition Results", result_text)
        else:
            QMessageBox.warning(self, "Warning", "No faces detected in the test image")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())