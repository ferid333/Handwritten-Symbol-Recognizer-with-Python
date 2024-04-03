from PyQt5.QtWidgets import QApplication,QMainWindow,QLabel,QPushButton,QMessageBox,QWidget,QVBoxLayout
from PyQt5.QtGui import QPainter,QColor,QPixmap,QFont,QPainter
from PIL import ImageOps,Image
import matplotlib.pyplot as plt
import sys
import numpy as np
import pickle
from model import nr_to_letter
class Win(QMainWindow):
    def __init__(self):
        super().__init__()
        self.loaded_model=pickle.load(open("model.pkl",'rb'))
        self.InitUI()
        self.show()
    def InitUI(self):
        self.widget=QWidget()
        self.setWindowTitle("Number Recognizer")
        self.setCentralWidget(self.widget)
        self.last_x, self.last_y = None, None
        self.label = QLabel()
        self.canvas=QPixmap(300,300)
        self.canvas.fill(QColor("black"))
        self.label.setPixmap(self.canvas)

        self.container =QVBoxLayout()
        self.container.setContentsMargins(0,0,0,0)

        self.prediction = QLabel('Prediction: ...')
        self.prediction.setFont(QFont('Monospace', 20))

        self.button_clear = QPushButton('CLEAR')
        self.button_clear.clicked.connect(self.clear_canvas)

        self.button_save = QPushButton('PREDICT')
        self.button_save.clicked.connect(self.predict)

        self.container.addWidget(self.label)
        self.container.addWidget(self.prediction)
        self.container.addWidget(self.button_clear)
        self.container.addWidget(self.button_save)
        self.widget.setLayout(self.container)

    def clear_canvas(self):
        self.canvas.fill(QColor('#000'))
        self.label.setPixmap(self.canvas)
        self.update()
    def predict(self):
        s=self.label.pixmap().toImage().bits().asarray(300 * 300 * 4)
        arr=np.frombuffer(s,dtype=np.uint8).reshape((300, 300, 4))
        arr =np.array(ImageOps.grayscale(Image.fromarray(arr).resize((28,28), Image.Resampling.LANCZOS)))
        # arr.save("imgggg.png")
        # plt.imshow(arr)
        # plt.show()
        arr=(arr/255.0).reshape(1,-1)
        self.prediction.setText("Prediction."+str(nr_to_letter[self.loaded_model.predict(arr)[0]]))
    def mouseMoveEvent(self, e):
        if self.last_x is None:
            self.last_x = e.x()
            self.last_y = e.y()
            return
        painter=QPainter(self.label.pixmap())
        pen=painter.pen()
        pen.setWidth(20)
        pen.setColor(QColor("white"))
        painter.setPen(pen)
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()

        self.last_x = e.x()
        self.last_y = e.y()
    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None    
app=QApplication(sys.argv)
win1=Win()
sys.exit(app.exec_())