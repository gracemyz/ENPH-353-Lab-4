#!/usr/bin/env python
#Shebang: it informs the shell that it should start the program listed (Python)
# and pass to it the contents of this file as a script

#importing libraries
from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi
#loadUi is for converting a QtDesigner.ui file to a Python object

import cv2
import sys
import numpy as np

# define the My_App class
class My_App(QtWidgets.QMainWindow):

	# constructor: loads the .ui file 
	def __init__(self):
		super(My_App, self).__init__()
		loadUi("./SIFT_app.ui", self)

		self._cam_id = 0
		self._cam_fps = 2
		self._is_cam_enabled = False
		self._is_template_loaded = False
		self._is_homography_on = False

		# connect signal/slots
		# when browse_button clicked, do the browse_button action
		self.browse_button.clicked.connect(self.SLOT_browse_button)
		self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)
		self.checkbox.stateChanged.connect(self.SLOT_checkbox)

		# make camera object 
		# set resolution to 320x240, low resolution
		self._camera_device = cv2.VideoCapture(self._cam_id)
		self._camera_device.set(3, 320)
		self._camera_device.set(4, 240)

		# Timer used to trigger the camera
		# emit signal every time set interval elapses
		self._timer = QtCore.QTimer(self)
		self._timer.timeout.connect(self.SLOT_query_camera)
		self._timer.setInterval(1000 / self._cam_fps)

		# Sift
		self.sift = cv2.xfeatures2d.SIFT_create()

	# constructor for slot function
	def SLOT_browse_button(self):

		#dlg: file dialog object
		dlg = QtWidgets.QFileDialog()
		#pick a file from existing files
		dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
		#if File Dialog object running, block app event loop?
		if dlg.exec_():
			# absolute path of selected file
			self.template_path = dlg.selectedFiles()[0]
		# image, for use in SIFT
		self.image_template = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
		# image, for use by GUI
		pixmap = QtGui.QPixmap(self.template_path)

		self.template_label.setPixmap(pixmap)
		# return absolute path of selected file
		print("Loaded template image file: " + self.template_path)

		# SIFT: find keypoints and features of the robot pic
		self.kp_image, self.desc_image = self.sift.detectAndCompute(self.image_template, None)
		# Feature matching
		index_params = dict(algorithm=0, trees=5)
		search_params = dict()
		self.flann = cv2.FlannBasedMatcher(index_params, search_params)

	def SLOT_checkbox(self):
		if self.checkbox.isChecked() == False:
			self._is_homography_on = False
			print("turned homography off")
		else:
			self._is_homography_on = True
			print("turned homography on")


	# Source: stackoverflow.com/questions/34232632/
	def convert_cv_to_pixmap(self, cv_img):
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		height, width, channel = cv_img.shape
		bytesPerLine = channel * width
		q_img = QtGui.QImage(cv_img.data, width, height, 
					 bytesPerLine, QtGui.QImage.Format_RGB888)
		return QtGui.QPixmap.fromImage(q_img)

	def SLOT_query_camera(self):
		ret, frame = self._camera_device.read()
		
		#TODO run SIFT on the captured frame
		grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		kp_grayframe, desc_grayframe = self.sift.detectAndCompute(grayframe, None)
		matches = self.flann.knnMatch(self.desc_image, desc_grayframe, k=2)


		good_points = []
		for m, n in matches:
			if m.distance < 0.6 * n.distance:
				good_points.append(m)

		if self._is_homography_on == False or len(good_points) < 10:
			img3 = cv2.drawMatches(self.image_template, self.kp_image, grayframe, kp_grayframe, good_points, grayframe)
			pixmap = self.convert_cv_to_pixmap(img3)
			self.live_image_label.setPixmap(pixmap)

		else:
			# homography!
			if len(good_points) > 10:
				query_pts = np.float32([self.kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
				train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
				matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
				matches_mask = mask.ravel().tolist()

				# Perspective transform
				h, w = (self.image_template).shape
				pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
				dst = cv2.perspectiveTransform(pts, matrix)

				homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
				pixmap = self.convert_cv_to_pixmap(frame)
				self.live_image_label.setPixmap(pixmap)

		

	def SLOT_toggle_camera(self):
		if self._is_cam_enabled:
			self._timer.stop()
			self._is_cam_enabled = False
			self.toggle_cam_button.setText("&Enable camera")
		else:
			self._timer.start()
			self._is_cam_enabled = True
			self.toggle_cam_button.setText("&Disable camera")


if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = My_App() #instantiate My_App class
	myApp.show()
	sys.exit(app.exec_()) #start Qt event loop and run until we exit
