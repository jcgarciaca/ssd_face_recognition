#!/usr/bin/env python
import rospy
import numpy as np
import dlib
import cv2
# from face_recognition.msg import Faces
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError

class FaceDetection(object):
	def __init__(self):
		"""Parameters"""
		self.rotations = rospy.get_param('~rotation_cycles', 0)
		self.rgb_camera = rospy.get_param('~rgb_camera', True)
		self.depth_camera = rospy.get_param('~depth_camera', False)
		self.ir_camera = rospy.get_param('~ir_camera', False)
		rgb_image_encoding = rospy.get_param('~rgb_image_encoding', "bgr8") #rgb8
		ir_image_encoding = rospy.get_param('~ir_image_encoding', "passthrougth") #mono8
		self.depth_image_encoding = rospy.get_param('~depth_image_encoding', "passthrougth") #mono8
		self.multiple_detection = rospy.get_param('~multiple_detection', True)
		self.show_detection = rospy.get_param('~show_detection', True)
		self.inform_detection = rospy.get_param('~inform_detection', False)
		self.min_area = rospy.get_param('~min_area',0)
		self.equalize_hist = rospy.get_param('~equalize_hist',True)
		"""Subscribers"""
		if self.rgb_camera:
			self.sub_bgr_image = rospy.Subscriber("/camera/color/image_raw",Image,self.callback_image)
			self.image_encoding = rgb_image_encoding
		elif self.ir_camera:
			self.sub_ir_image = rospy.Subscriber("ir_image",Image,self.callback_image)
			self.image_encoding = ir_image_encoding

		if self.depth_camera:
			self.sub_bgr_image = rospy.Subscriber("depth_image",Image,self.callback_depth_image)
		"""Publishers"""
		self.pub_face_image = rospy.Publisher("face_recognition_tp",Image,queue_size = 1)
		if self.show_detection:
			self.pub_detections = rospy.Publisher("detected_faces",Image,queue_size = 1)
		if self.inform_detection:
			self.pub_flag = rospy.Publisher("any_detection",Bool,queue_size = 1)
		"""Node Configuration"""
		self.image = None
		self.depth_image = None
		self.image_shape = None #[h,w]
		self.bridge = CvBridge()
		self.face_detector = dlib.get_frontal_face_detector()

		self.main()

	def callback_image(self,msg):
		if self.image is None:
			try:
				image = self.bridge.imgmsg_to_cv2(msg, self.image_encoding)
			except CvBridgeError as e:
				print(e)
			image = np.rot90(image, k=self.rotations)

			if self.image_shape is None:
				self.image_shape = image.shape
			self.image = image

	def callback_depth_image(self,msg):
		if self.depth_image is None:
			try:
				image = self.bridge.imgmsg_to_cv2(msg, self.depth_image_encoding)
			except CvBridgeError as e:
				print(e)
			self.depth_image = np.rot90(image, k=self.rotations)

	def crop_images(self,image,rect,expand=0.4):
		v_dist = int(abs(rect.bottom() - rect.top())*expand/2)
		h_dist = int(abs(rect.right() - rect.left())*expand/2)
		crop_img = image[max(0, rect.top()-v_dist): min(rect.bottom()+v_dist, self.image_shape[0]),
							 max(0, rect.left()-h_dist): min(rect.right()+h_dist, self.image_shape[1])]

		if self.equalize_hist:
			if self.rgb_camera:
				img_yuv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2YUV)
				img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
				crop_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
			elif self.ir_camera:
				crop_img = cv2.equalizeHist(crop_img)

		if self.depth_camera:
			crop_depth_img = self.depth_image[max(0, rect.top()): min(rect.bottom(), self.image_shape[0]),
											  max(0, rect.left()): min(rect.right(), self.image_shape[1])]
		else:
			crop_depth_img = None
		return crop_img,crop_depth_img

	def main(self):
		detected_faces = [] # Faces()
		imagen = Image()
		while not rospy.is_shutdown():
			if not(self.image is None) and (not(self.depth_camera) or not(self.depth_image is None)):
				detected_faces = []
				# detected_faces.faces_depth_images = []
				self.pub_detections.publish(self.bridge.cv2_to_imgmsg(self.image,self.image_encoding))
				
				rects = self.face_detector(self.image, 0)

				if len(rects) == 0:
					people_detected = False
				else:
					people_detected = True
				if self.inform_detection:
					self.pub_flag.publish(people_detected)
				if not(self.multiple_detection):
					closest_rect = None

				for rect in rects:
					if rect.area() >= self.min_area:
						if self.multiple_detection:
							crop_img,crop_depth_img = self.crop_images(self.image,rect)
							try:
								detected_faces.append(self.bridge.cv2_to_imgmsg(crop_img, self.image_encoding))
								if self.depth_camera:
									detected_faces.faces_depth_images.append(self.bridge.cv2_to_imgmsg(crop_depth_img,self.depth_image_encoding))
							except CvBridgeError as e:
								print(e)
							if self.show_detection:
								cv2.rectangle(self.image, (rect.left(),rect.top()), (rect.right(),rect.bottom()), (0,255,0),2)
						else:
							if closest_rect is None:
								closest_rect = rect
							else:
								if self.depth_camera:
									roi_depth_img = self.depth_image[max(0, rect.top()): min(rect.bottom(), self.image_shape[0]),
																	  max(0, rect.left()): min(rect.right(), self.image_shape[1])]
									closest_rect_roi_depth_img = self.depth_image[max(0, closest_rect.top()): min(closest_rect.bottom(), self.image_shape[0]),
																				  max(0, closest_rect.left()): min(closest_rect.right(), self.image_shape[1])]
									if cv2.mean(roi_depth_img) < cv2.mean(closest_rect_roi_depth_img):
										closest_rect = rect
								else:
									if rect.area() > closest_rect.area():
										closest_rect = rect

				if people_detected:
					if not(self.multiple_detection):
						crop_img,crop_depth_img = self.crop_images(self.image,closest_rect)

						try:
							detected_faces.append(self.bridge.cv2_to_imgmsg(crop_img, self.image_encoding))
							if self.depth_camera:
								detected_faces.faces_depth_images.append(self.bridge.cv2_to_imgmsg(crop_depth_img,self.depth_image_encoding))
						except CvBridgeError as e:
							print(e)

						if self.show_detection:
							cv2.rectangle(self.image, (closest_rect.left(),closest_rect.top()), (closest_rect.right(),closest_rect.bottom()), (0,255,0),2)

					imagen = detected_faces[0]
					self.pub_face_image.publish(imagen)

					if self.show_detection:
						self.pub_detections.publish(self.bridge.cv2_to_imgmsg(self.image,self.image_encoding))

				self.image = self.depth_image = None

if __name__ == '__main__':
	# import rospkg
	try:
		rospy.init_node("face_detection", anonymous = True)
		# rospack = rospkg.RosPack()
		# PACK_PATH = rospack.get_path('face_recognition') + "/scripts/"
		face_detection = FaceDetection()
	except KeyboardInterrupt:
		print("Shutting down")
	cv2.destroyAllWindows()
