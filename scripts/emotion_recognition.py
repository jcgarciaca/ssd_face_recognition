#!/usr/bin/env python
import rospy
import sys
import os
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

# keras imports
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import PIL
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.preprocessing import image


class EmotionRecognition():
    def __init__(self):        
        self.root_path = os.path.join(os.path.dirname(sys.path[0]))
        self.emotion_model_path = os.path.join(self.root_path, 'models', 'emotion_models', 'fer2013_mini_XCEPTION.102-0.66.hdf5')
        self.gender_model_path = os.path.join(self.root_path, 'models', 'gender_models', 'simple_CNN.81-0.96.hdf5')
        self.emotion_labels = self.get_labels('fer2013')
        self.gender_labels = self.get_labels('imdb')

        self.emotion_classifier = load_model(self.emotion_model_path, compile=False)
        self.gender_classifier = load_model(self.gender_model_path, compile=False)

        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]
        self.gender_target_size = self.gender_classifier.input_shape[1:3]

        # self.gender_classifier.summary()

        self.bridge = CvBridge()
        # self.img_path = os.path.join(self.root_path, 'img_received.jpg')
        self.enable = False
        self.available = True

        # Subscriber
        rospy.Subscriber('/face_recognition_tp', Image, self.analyze_images, queue_size = 1)

        # Publisher
        # self.name_pub = rospy.Publisher('/detected_person_name', String, queue_size = 1)

    
    def get_labels(self, dataset_name):
        if dataset_name == 'fer2013':
            return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
        elif dataset_name == 'imdb':
            return {0: 'woman', 1: 'man'}
        elif dataset_name == 'KDEF':
            return {0: 'AN', 1: 'DI', 2: 'AF', 3: 'HA', 4: 'SA', 5: 'SU', 6: 'NE'}
        else:
            raise Exception('Invalid dataset name')

    def preprocess_input(self, x, v2=True):
        x = x.astype('float32')
        x = x / 255.0
        if v2:
            x = x - 0.5
            x = x * 2.0
        return x
    

    def analyze_images(self, msg):
        if self.available:
            # convert to opencv
            # print('receives image')
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            except CvBridgeError as e:
                print(e)
            rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            self.rgb_face = np.asarray(rgb)
            self.gray_face = np.asarray(gray)
            self.gray_face = np.squeeze(self.gray_face)
            self.gray_face = self.gray_face.astype('uint8')
            
            try:
                self.rgb_face = cv2.resize(self.rgb_face, (self.gender_target_size))
                self.gray_face = cv2.resize(self.gray_face, (self.emotion_target_size))
                self.enable = True
            except:
                self.enable = False
          

    def run(self):
        while not rospy.is_shutdown():
            if self.enable:
                self.available = False
                self.enable = False
                
                self.rgb_face = self.preprocess_input(self.rgb_face, False)
                self.rgb_face = np.expand_dims(self.rgb_face, 0)

                self.gray_face = self.preprocess_input(self.gray_face, True)
                self.gray_face = np.expand_dims(self.gray_face, 0)
                self.gray_face = np.expand_dims(self.gray_face, -1)
                
                                
                gender_prediction = self.gender_classifier.predict(self.rgb_face)
                gender_label_arg = np.argmax(gender_prediction)
                gender_text = self.gender_labels[gender_label_arg]

                emotion_label_arg = np.argmax(self.emotion_classifier.predict(self.gray_face))
                emotion_text = self.emotion_labels[emotion_label_arg]
                
                print(gender_text, emotion_text)
                self.available = True


if __name__ == "__main__":
    rospy.init_node('emotion_recognition_node', anonymous = True)
    rospy.loginfo('emotion_recognition_node started')
    emotion_recognition = EmotionRecognition()
    emotion_recognition.run()
    rospy.spin()
