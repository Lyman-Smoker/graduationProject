# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/liym/Desktop/Code/PYQT5_learning/GUI2.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.widget = QVideoWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(10, 200, 400, 400))
        self.widget.setObjectName("widget")
        MainWindow.setCentralWidget(self.centralwidget)

        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")

        self.formLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(500, 220, 261, 61))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.pushButton_2 = QtWidgets.QPushButton(self.formLayoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.pushButton_2)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.lineEdit_2)
        self.btn_confirm = QtWidgets.QPushButton(self.centralwidget)
        self.btn_confirm.setGeometry(QtCore.QRect(580, 290, 113, 32))
        self.btn_confirm.setObjectName("pushButton")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(500, 370, 291, 131))
        self.textEdit.setObjectName("textEdit")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(340, 30, 111, 16))
        self.label.setObjectName("label")

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "逗比程序"))
        self.pushButton_2.setText(_translate("MainWindow", "选择视频"))
        self.btn_confirm.setText(_translate("MainWindow", "评估"))
        # self.label.setText(_translate("MainWindow", "动作质量评估系统"))

from PyQt5.QtMultimediaWidgets import QVideoWidget

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())