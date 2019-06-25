import numpy as np
import cv2
import os
import dlib

def crop_images(image,rect,expand=0.4):
    v_dist = int(abs(rect.bottom() - rect.top())*expand/2)
    h_dist = int(abs(rect.right() - rect.left())*expand/2)

    crop_img = image[max(0, rect.top()-v_dist): min(rect.bottom()+v_dist, image_shape[0]),
                         max(0, rect.left()-h_dist): min(rect.right()+h_dist, image_shape[1])]

    img_yuv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    crop_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return crop_img


files = os.listdir("./")
face_detector = dlib.get_frontal_face_detector()
for file in files:
    if file.endswith(".jpg"):
        print("loading image with name {}".format(file))
        img = cv2.imread(file)
        image_shape = img.shape
        rects = face_detector(img,0)
        # for rect in rects:
        face = crop_images(img,rects[0])
        cv2.imshow('Main', face)
        cv2.waitKey(1)
        x = raw_input(">>>")
        #    if x == 'e':
        #        break
        # img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        cv2.imwrite('faces_exp/'+file,face)
