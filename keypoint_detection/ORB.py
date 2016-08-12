import cv2

from keypoint_detection.keypoint_pairs import KeypointList

__author__ = 'Dandi Chen'


class ORB_point(KeypointList):
    def __init__(self, bbox):
        KeypointList.__init__(self, bbox)

    def get_keypoint(self, img):
        # Initiate ORB detector
        orb = cv2.ORB_create()
        orb.setMaxFeatures(1200)
        orb.setScaleFactor(1.25)
        orb.setNLevels(6)
        orb.setEdgeThreshold(10)
        orb.setPatchSize(20)
        orb.setFastThreshold(8)

        # find the keypoints with ORB
        kp = orb.detect(img, None)

        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)

        self.init_val(kp, des)





