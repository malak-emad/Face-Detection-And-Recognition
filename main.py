import sys
import os
import argparse
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
                            QTabWidget, QTextEdit, QProgressBar, QComboBox, QSpinBox, QDoubleSpinBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import logging
logger = logging.getLogger(__name__)
from face_detection import FaceDetection




# Load the UI file
ui, _ = loadUiType("faceDetRecUI.ui")

class MainApp(QtWidgets.QMainWindow, ui):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)

        self.image = None

        self.upload_button.clicked.connect(self.upload)
        self.applyDetection_button.clicked.connect(self.apply_face_detection)


    def upload(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.webp);;All Files (*)", options=options)

        if file_path:
            self.q_image, self.image = self.process_and_store_image(file_path)  
            self.inputDetection.setPixmap(QPixmap.fromImage(self.q_image))
            self.outputDetection.setPixmap(QPixmap())

            #set scaled contents for each QLabel only once
            self.inputDetection.setScaledContents(True)
            self.outputDetection.setScaledContents(True)
        print("upload")
            

    def process_and_store_image(self, file_path):
        original_image = Image.open(file_path).convert("RGB")
        self.img_array = np.array(original_image)
    
        #convert PIL image to QImage
        height, width, channels = self.img_array.shape
        bytes_per_line = channels * width
        q_image = QImage(self.img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        return q_image, self.img_array
    

    def apply_face_detection(self):
        rectangle_tickness = int(self.tickness_lineEdit.text())
        scale_factor = float(self.scaleFactor_lineEdit.text())
        window_size = int(self.windowSize_lineEdit.text())
        
        faces, detected_faces = FaceDetection.detect_faces(self.img_array, rectangle_tickness, scale_factor, window_size)
        self.display_result_on_label(self.outputDetection, detected_faces)


    def display_result_on_label(self, label: QLabel, image: np.ndarray):
        """
        Converts a NumPy array (grayscale or RGB) to QPixmap and sets it on a QLabel.
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Handle grayscale and color separately
        if len(image.shape) == 2:  # Grayscale
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        elif len(image.shape) == 3 and image.shape[2] == 3:  # Color
            height, width, channels = image.shape
            bytes_per_line = channels * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            raise ValueError("Unsupported image format for displaying.")

        pixmap = QPixmap.fromImage(q_image)

        # Resize pixmap nicely
        pixmap = pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        label.setPixmap(pixmap)
        label.setScaledContents(True)






if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())