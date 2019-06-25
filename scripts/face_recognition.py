#!/usr/bin/env python
import rospy
import sys
import os
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

# keras imports
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import PIL
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image


class FaceRecognition():
    def __init__(self):        
        self.root_path = os.path.join(os.path.dirname(sys.path[0]))
        self.model_path = os.path.join(self.root_path, 'models', 'vgg_face_weights.h5')
        self.model = self.create_model()
        self.model.load_weights(self.model_path)
        self.model._make_predict_function()
        self.vgg_face_descriptor = Model(inputs=self.model.layers[0].input, outputs=self.model.layers[-2].output)
        self.threshold = 0.19 #0.22 # 0.2419
        self.npy_folder = os.path.join(self.root_path, 'imgs_ref_npy')
        self.person_descriptors = np.load(self.npy_folder + '/' + 'persons.npy')
        self.f = open(self.npy_folder + '/' + 'list_names.txt', 'r')
        self.data_names = self.f.read().split(',')
        self.bridge = CvBridge()
        self.img_path = os.path.join(self.root_path, 'img_received.jpg')
        self.enable = False
        self.available = True
        
        self.message_name = String()
        self.previos_index = None

        # Subscriber
        rospy.Subscriber('/face_recognition_tp', Image, self.analyze_images, queue_size = 1)

        # Publisher
        self.name_pub = rospy.Publisher('/detected_person_name', String, queue_size = 1)


    def create_model(self):
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Convolution2D(4096, (7, 7), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation('softmax'))
        return model

    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img)        
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    def findCosineSimilarity(self, source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def getRepresentation(self):
        img_representation = self.vgg_face_descriptor.predict(self.preprocess_image(self.img_path))[0,:]
        return img_representation

    def check_representation(self, img1_representation):
        for index, descriptor in enumerate(self.person_descriptors):
            cosine_similarity = self.findCosineSimilarity(img1_representation, descriptor)
            
            if(cosine_similarity < self.threshold):
                return [True, index, cosine_similarity]
        return [False, len(descriptor) + 1, 1.0]

    def analyze_images(self, msg):
        if self.available:
            # convert to opencv
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            except CvBridgeError as e:
                print(e)
            image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)        

            cv2.imwrite(self.img_path, cv_image)
            rospy.sleep(0.1)
            self.enable = True
          

    def run(self):
        while not rospy.is_shutdown():
            if self.enable:
                self.available = False
                self.enable = False
                im_rep = self.getRepresentation()

                [flag, index_match, cosine_similarity] = self.check_representation(im_rep)
                # edward,john,felipe,juanc,juanf,jorge,david,philippe,cristian,andresz,daniel
                if flag:
                    print(self.data_names[index_match] + ', ' + str(cosine_similarity))
                    self.message_name.data = self.data_names[index_match]
                    if self.data_names[index_match] == 'juanferro':
                        self.message_name.data = 'chulo'
                    elif self.data_names[index_match] == 'cristian':
                        self.message_name.data = 'iphone'
                    elif self.data_names[index_match] == 'chavez':
                        self.message_name.data = 'papa'
                    if self.previos_index != index_match:
                        self.previos_index = index_match
                        self.name_pub.publish(self.message_name)
                else:
                    print('NO MATCH')
                self.available = True
            rospy.sleep(0.1)



if __name__ == "__main__":
    rospy.init_node('face_recognition_node', anonymous = True)
    rospy.loginfo('face_recognition_node started')
    face_recognition = FaceRecognition()
    face_recognition.run()
    rospy.spin()
