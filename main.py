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




# Load the UI file
ui, _ = loadUiType("faceDetRecUI.ui")

class MainApp(QtWidgets.QMainWindow, ui):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)





if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())