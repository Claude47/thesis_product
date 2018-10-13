"""
    Claude Betz (BTZCLA001)
    imageView.py

    Subclasses the QLabel class to add drag and drop functionality for a User to
    select a region of interest.
"""

from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, qApp, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QRubberBand
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import Qt, QRect, pyqtSlot, pyqtSignal, QSize 

class imageView(QLabel):
    """image view subclasses QLabel to render and add mouse events"""
    currentQRect = 0
    def __init__(self):
        super(imageView, self).__init__()
        self.pixmap = QPixmap() # to store image
        self.cropLabel = QLabel() # user selected crop

    # Mouse Events
    def mousePressEvent(self, eventQMouseEvent):
        self.originQPoint = eventQMouseEvent.pos()
        self.currentQRubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.currentQRubberBand.setGeometry(QRect(self.originQPoint, QSize()))
        self.currentQRubberBand.show()

    def mouseMoveEvent(self, eventQMouseEvent):
        self.currentQRubberBand.setGeometry(QRect(self.originQPoint, eventQMouseEvent.pos()).normalized())

    def mouseReleaseEvent(self, eventQMouseEvent):
        #self.currentQRubberBand.hide()
        QRect = self.currentQRubberBand.geometry()
        self.currentQRect = (QRect.y(), QRect.x(), QRect.height(), QRect.width())
        #print(self.currentQRect)
        self.currentQRubberBand.deleteLater()
        cropPixmap = self.pixmap.copy(QRect)
        self.cropLabel.setPixmap(cropPixmap)
        # cropPixmap.save('output.png')

