#!/usr/bin/env python
import rospy
import sys
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class TakeImage:
    def __init__(self):
        self.root_path = os.path.join(os.path.dirname(sys.path[0]))
        self.imgs_db = os.path.join(self.root_path, 'images_db')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_cb, queue_size=1)
    
    def image_cb(self, data):
        # rospy.loginfo("received image")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        cv2.imshow("cv_image", cv_image)
        
        k = cv2.waitKey(1)
        if k % 256 == 115:
            name = input('Enter name: ')
            cv2.imwrite(os.path.join(self.imgs_db, name + '.jpg'), cv_image)
            print("image saved")     

def main(args):
    rospy.init_node('save_face_node', anonymous = True)
    rospy.loginfo('save_face_node started')
    img = TakeImage()
    rospy.loginfo('Running')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("ShutDown")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
