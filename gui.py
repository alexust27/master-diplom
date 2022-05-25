import sys

from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap
from PyQt5 import uic, QtCore

from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication, QComboBox

from heatmap_compare import HeatMap

images_formats = '*.jpg;*.jpeg;*.png'


class HeatmapGui(QMainWindow):

    def __init__(self):
        super().__init__()
        uic.loadUi('main.ui', self)
        self.__init_ui()
        self.file_name = ""
        self.file_name2 = ""
        self.file_name3 = ""
        self.__method_id = 0

    def __init_ui(self):
        self.btn_open1 = self.findChild(QPushButton, "open_1")
        self.btn_open2 = self.findChild(QPushButton, "open_2")
        self.btn_compare = self.findChild(QPushButton, "compare_b")

        self.methods_box = self.findChild(QComboBox, "comboBox")
        self.methods_box.currentIndexChanged.connect(self.selection_change)

        self.out_1 = self.findChild(QLabel, "label")
        self.out_2 = self.findChild(QLabel, "label_2")
        self.out_3 = self.findChild(QLabel, "label_3")

        self.btn_open1.clicked.connect(self.buttonClicked)
        self.btn_open2.clicked.connect(self.buttonClicked2)
        self.btn_compare.clicked.connect(self.button_compare)

        self.show()

    def buttonClicked(self):
        file_name = QFileDialog.getOpenFileName(filter=images_formats)[0]
        if file_name == "":
            return
        self.file_name = file_name

        print(self.file_name)

        pixmap = QPixmap(self.file_name)
        pixmap_resized = pixmap.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
        self.out_1.setPixmap(pixmap_resized)
        self.out_1.resize(pixmap_resized.width(), pixmap_resized.height())

        self.show()

    def buttonClicked2(self):
        file_name = QFileDialog.getOpenFileName(filter=images_formats)[0]
        if file_name == "":
            return
        self.file_name2 = file_name

        print(self.file_name2)

        pixmap = QPixmap(self.file_name2)
        pixmap_resized = pixmap.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
        self.out_2.setPixmap(pixmap_resized)
        self.out_2.resize(pixmap_resized.width(), pixmap_resized.height())

        self.show()

    def selection_change(self, i):
        self.__method_id = i
        print("method", i + 1)

    def button_compare(self):
        if self.file_name == "" or self.file_name2 == "":
            return
        hm = HeatMap(self.file_name, self.file_name2, self.__method_id, use_gui=True)
        hm.create()

        pixmap = QPixmap(hm.get_path())
        pixmap = pixmap.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
        self.out_3.setPixmap(pixmap)
        self.out_3.resize(pixmap.width(), pixmap.height())

        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    g = HeatmapGui()
    sys.exit(app.exec_())
