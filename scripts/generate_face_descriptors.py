#!/usr/bin/env python
import rospy
import sys
import os

# keras imports
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import PIL
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image


class GenerateDescriptors():
    def __init__(self):        
        self.root_path = os.path.join(os.path.dirname(sys.path[0]))
        self.model_path = os.path.join(self.root_path, 'models', 'vgg_face_weights.h5')
        self.model = self.create_model()
        self.model.load_weights(self.model_path)
        self.model._make_predict_function()
        self.vgg_face_descriptor = Model(inputs=self.model.layers[0].input, outputs=self.model.layers[-2].output)
        self.npy_file = os.path.join(self.root_path, 'imgs_ref_npy', 'persons.npy')
        self.names_list_file = os.path.join(self.root_path, 'imgs_ref_npy', 'list_names.txt')
        self.imgs_folder = os.path.join(self.root_path, 'images_db')
        


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

    def getRepresentation(self, image_path):
        img_representation = self.vgg_face_descriptor.predict(self.preprocess_image(image_path))[0,:]
        return img_representation

    def check_representation(self, img1_representation):
        for index, descriptor in enumerate(self.person_descriptors):
            cosine_similarity = self.findCosineSimilarity(img1_representation, descriptor)
            
            if(cosine_similarity < self.threshold):
                return [True, index, cosine_similarity]
        return [False, len(descriptor) + 1, 1.0]

    def analyze_images(self, msg):
        # convert to opencv        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            print(e)
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)        

        cv2.imwrite(self.img_path, image)
        rospy.sleep(0.1)
        self.enable = True
          

    def run(self):
        imgs_list = os.listdir(self.imgs_folder)
        person_names = []
        person_rep = []

        for index, person in enumerate(imgs_list):
            person_names.append(person.split('.')[0])
            im_rep = self.getRepresentation(os.path.join(self.imgs_folder, person))
            person_rep.append(im_rep)
        np.save(self.npy_file, person_rep)
        print('shape:', len(person_rep), person_rep[0].shape)

        person_descriptors = np.load(self.npy_file)
        print('loaded!! shape:', len(person_descriptors), person_descriptors[0].shape)

        with open(self.names_list_file, 'w+') as f:
            for enum, item in enumerate(person_names):
                if enum == len(person_names) - 1:
                    f.write("%s" % item)
                else:
                    f.write("%s," % item)
        print('Completed')

if __name__ == "__main__":
    rospy.init_node('generate_face_descriptors_node', anonymous = True)
    rospy.loginfo('generate_face_descriptors_node started')
    generate_descriptors = GenerateDescriptors()
    generate_descriptors.run()
    rospy.spin()
