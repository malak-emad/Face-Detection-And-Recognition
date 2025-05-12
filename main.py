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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the UI file
ui, _ = loadUiType("faceDetRecUI.ui")

class MainApp(QtWidgets.QMainWindow, ui):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)

        # Initialize attributes
        self.image = None
        self.face_recognition = FaceRecognition()
        self.test_images = []
        self.test_image_paths = []
        self.current_test_image_index = -1

        # Connect buttons to their functions
        self.upload_button.clicked.connect(self.upload)
        self.applyDetection_button.clicked.connect(self.apply_face_detection)
        
        # Connect new buttons
        self.uploadImage1_button.clicked.connect(self.upload_training_data)
        self.uploadImage2_button.clicked.connect(self.make_eigen_faces)
        self.pushButton.clicked.connect(self.load_testing_database)
        self.applyMethods_button.clicked.connect(self.match_faces)
        
        # Status display
        self.statusBar().showMessage("Ready")

    def upload(self):
        """Upload a single image for face detection"""
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

            # Set scaled contents for each QLabel
            self.inputDetection.setScaledContents(True)
            self.outputDetection.setScaledContents(True)
            
            self.statusBar().showMessage(f"Loaded image: {os.path.basename(file_path)}")
            logger.info(f"Uploaded image from {file_path}")

    def process_and_store_image(self, file_path):
        """Process an image and convert it to format usable by Qt"""
        original_image = Image.open(file_path).convert("RGB")
        img_array = np.array(original_image)
    
        # Convert PIL image to QImage
        height, width, channels = img_array.shape
        bytes_per_line = channels * width
        q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        return q_image, img_array
    
    def apply_face_detection(self):
        """Apply face detection to the currently loaded image"""
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please upload an image first.")
            return
            
        try:
            rectangle_thickness = int(self.tickness_lineEdit.text())
            scale_factor = float(self.scaleFactor_lineEdit.text())
            window_size = int(self.windowSize_lineEdit.text())
            
            start = time.time()
            faces, detected_faces = FaceDetection.detect_faces(self.image, rectangle_thickness, scale_factor, window_size)
            end = time.time()
            
            self.display_result_on_label(self.outputDetection, detected_faces)
            elapsed_ms = (end - start) * 1000  # Convert to milliseconds
            self.detectionTime_label.setText(f"{elapsed_ms:.2f} ms")
            
            self.statusBar().showMessage(f"Detected {len(faces)} faces in {elapsed_ms:.2f} ms")
            logger.info(f"Face detection completed. Found {len(faces)} faces.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during face detection: {str(e)}")
            logger.error(f"Face detection error: {str(e)}")

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

    def upload_training_data(self):
        """Upload training data for face recognition"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Training Data Folder")
        
        if not folder_path:
            self.statusBar().showMessage("Training data upload cancelled")
            return
            
        try:
            # Use new method to upload data
            success, message, images_data = self.face_recognition.load_training_data(folder_path)
            
            if success:
                self.statusBar().showMessage(message)
                # Enable other buttons after training data is loaded
                self.uploadImage2_button.setEnabled(True)
                
                # Visualize the training data
                self.face_recognition.visualize_training_data(images_data)
            else:
                QMessageBox.warning(self, "Warning", message)
                self.statusBar().showMessage("Failed to load training data")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error uploading training data: {str(e)}")
            logger.error(f"Training data upload error: {str(e)}")

    def make_eigen_faces(self):
        """Generate eigenfaces using PCA"""
        if not hasattr(self.face_recognition, 'training_images') or len(self.face_recognition.training_images) == 0:
            QMessageBox.warning(self, "Warning", "Please upload training data first.")
            return
            
        try:
            success, message = self.face_recognition.compute_eigenfaces()
            
            if success:
                self.statusBar().showMessage(message)
                # Enable test data loading after eigenfaces are computed
                self.pushButton.setEnabled(True)
            else:
                QMessageBox.warning(self, "Warning", message)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating eigenfaces: {str(e)}")
            logger.error(f"Eigenfaces generation error: {str(e)}")

    def load_testing_database(self):
        """Load test images for face recognition"""
        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Test Images", "", 
            "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.webp *.pgm);;All Files (*)", 
            options=options
        )
        
        if not file_paths:
            self.statusBar().showMessage("Test image loading cancelled")
            return
            
        try:
            self.test_images = []
            self.test_image_paths = file_paths
            
            for file_path in file_paths:
                img = cv2.imread(file_path)
                if img is not None:
                    self.test_images.append(img)
            
            if self.test_images:
                self.current_test_image_index = 0
                # Display the first test image
                self.display_current_test_image()
                self.applyMethods_button.setEnabled(True)
                self.statusBar().showMessage(f"Loaded {len(self.test_images)} test images")
            else:
                QMessageBox.warning(self, "Warning", "No valid images were loaded.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading test images: {str(e)}")
            logger.error(f"Test image loading error: {str(e)}")

    def display_current_test_image(self):
        """Display the current test image"""
        if 0 <= self.current_test_image_index < len(self.test_images):
            img = self.test_images[self.current_test_image_index]
            
            # Convert to RGB for Qt display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channels = img_rgb.shape
            bytes_per_line = channels * width
            q_image = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Display in BOTH input labels for consistency
            self.inputDetection.setPixmap(QPixmap.fromImage(q_image))
            self.inputDetection.setScaledContents(True)
            # NEW: Also display in inputRecognition
            self.inputRecognition.setPixmap(QPixmap.fromImage(q_image))
            self.inputRecognition.setScaledContents(True)
            
            # Clear output labels
            self.outputDetection.setPixmap(QPixmap())
            self.outputRecognition.setPixmap(QPixmap())
            
            # Store current image for processing
            self.image = img_rgb
                
                # Show current image info
            file_name = os.path.basename(self.test_image_paths[self.current_test_image_index])
            self.statusBar().showMessage(f"Test image {self.current_test_image_index + 1}/{len(self.test_images)}: {file_name}")

    def match_faces(self):
        """Perform face recognition on the current test image"""
        if self.image is None:
            QMessageBox.warning(self, "Warning", "No test image loaded.")
            return

        if not hasattr(self.face_recognition, 'eigenfaces') or self.face_recognition.eigenfaces is None:
            QMessageBox.warning(self, "Warning", "Please generate eigenfaces first.")
            return

        try:
            # Convert RGB to BGR for OpenCV processing if needed
            if len(self.image.shape) == 3 and self.image.shape[2] == 3:
                img_for_recognition = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            else:
                img_for_recognition = self.image.copy()

            # Perform face recognition
            recognized_image, results, best_match_image = self.face_recognition.recognize_face(img_for_recognition)

            if recognized_image is not None:
                # Convert recognized image back to RGB for display
                if len(recognized_image.shape) == 3 and recognized_image.shape[2] == 3:
                    recognized_image_rgb = cv2.cvtColor(recognized_image, cv2.COLOR_BGR2RGB)
                else:
                    recognized_image_rgb = recognized_image

                # Prepare training match image
                if best_match_image is not None:
                    if len(best_match_image.shape) == 2:
                        best_match_image_rgb = cv2.cvtColor(best_match_image, cv2.COLOR_GRAY2RGB)
                    else:
                        best_match_image_rgb = cv2.cvtColor(best_match_image, cv2.COLOR_BGR2RGB)
                else:
                    best_match_image_rgb = None

                # Show test image with rectangles on left label
                self.display_result_on_label(self.outputDetection, recognized_image_rgb)

                # Show training match image on right label (unmodified)
                if best_match_image_rgb is not None:
                    self.display_result_on_label(self.outputRecognition, best_match_image_rgb)

                # Show recognition results
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

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during face recognition: {str(e)}")
            logger.error(f"Face recognition error: {str(e)}")



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())