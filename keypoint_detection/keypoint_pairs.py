import cv2
import numpy as np

__author__ = 'Dandi Chen'


def read_img_pair(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    return img1, img2


class KeypointList(object):
    def __init__(self, bbox, kp=None, pt_x=None, pt_y=None, pos_x=None, pos_y=None, des=None, length=0):
        self.bbox = bbox         # focus on only one bounding box in each frame
        self.kp = kp
        self.pt_x = pt_x
        self.pt_y = pt_y
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.des = des
        self.length = length

    def init_val(self, kp, des):
        self.length = len(kp)
        self.kp = kp
        self.des = des
        self.pt_x = np.zeros(self.length)
        self.pt_y = np.zeros(self.length)
        self.pos_x = np.zeros(self.length)
        self.pos_y = np.zeros(self.length)

        for idx in range(self.length):
            (x, y) = self.kp[idx].pt
            self.pt_x[idx] = x + self.bbox.top_left_x
            self.pt_y[idx] = y + self.bbox.top_left_y

        self.pos_x = (self.pt_x - self.bbox.top_left_x) / self.bbox.width
        self.pos_y = (self.pt_y - self.bbox.top_left_y) / self.bbox.height

    def set_val(self, idx):
        self.length = len(idx)
        tmp = []
        for i in np.asarray(idx)[0]:
            tmp.append(self.kp[i])
        self.kp = None
        self.kp = tmp

        self.pt_x = self.pt_x[idx]
        self.pt_y = self.pt_y[idx]
        self.pos_x = self.pos_x[idx]
        self.pos_y = self.pos_y[idx]
        self.des = self.des[idx]


class KeypointPairList(object):
    def __init__(self, list1, list2):
        self.list1 = list1
        self.list2 = list2
        self.neighbor_mat = np.zeros((self.list1.length, self.list2.length), dtype=bool)
        self.distance = np.zeros(min(self.list1.length, self.list2.length), dtype=float)

    def init_val(self, kp1, kp2, des1, des2):
        self.list1.init_val(kp1, des1)
        self.list2.init_val(kp2, des2)

    def set_val(self, idx):
        self.list1.set_val(idx)
        self.list2.set_val(idx)
        self.neighbor_mat = self.neighbor_mat[idx]
        self.distance = self.distance[idx]

    def get_euclidean_dis(self):
        pt1 = np.array([self.list1.pos_x, self.list1.pos_y]).transpose()
        pt2 = np.array([self.list2.pos_x, self.list2.pos_y]).transpose()
        self.distance = np.sqrt(np.sum((pt1 - pt2) ** 2, axis=1))

    def vis_pt_pairs(self, img1, img2):
        shown_img1 = img1
        shown_img2 = img2

        if self.list1.bbox.top_left_x == 0 and self.list1.bbox.top_left_y == 0 \
                and self.list2.bbox.top_left_x == 0 and self.list2.bbox.top_left_y == 0:
            shown_img1 = cv2.drawKeypoints(img1, self.list1.kp, shown_img1, color=(0, 255, 0), flags=0)
            cv2.imshow('image1', shown_img1)
            cv2.waitKey(0)

            shown_img2 = cv2.drawKeypoints(img2, self.list2.kp, shown_img2, color=(0, 255, 0), flags=0)
            cv2.imshow('image2', shown_img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            if self.list1.bbox.top_left_x != 0 or self.list1.bbox.top_left_y != 0:
                for idx in range(self.list1.length):
                    cv2.circle(shown_img1, (int(self.list1.pt_x[idx] + self.list1.bbox.top_left_x),
                                            int(self.list1.pt_y[idx] + self.list1.bbox.top_left_y)),
                               3, color=(0, 255, 0))
                cv2.imshow('image1', shown_img1)
                cv2.waitKey(0)

            if self.list2.bbox.top_left_x != 0 or self.list2.bbox.top_left_y != 0:
                for idx in range(self.list2.length):
                    cv2.circle(shown_img2, (int(self.list2.pt_x[idx] + self.list2.bbox.top_left_x),
                                            int(self.list2.pt_y[idx] + self.list2.bbox.top_left_y)),
                               3, color=(0, 255, 0))
                cv2.imshow('image2', shown_img2)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def write_pt_pairs(self, img1, img2, kp_path1, kp_path2):
        shown_img1 = img1
        shown_img2 = img2

        if self.list1.bbox.top_left_x == 0 and self.list1.bbox.top_left_y == 0 \
                and self.list2.bbox.top_left_x == 0 and self.list2.bbox.top_left_y == 0:
            shown_img1 = cv2.drawKeypoints(img1, self.list1.kp, shown_img1, color=(0, 255, 0), flags=0)
            cv2.imwrite(kp_path1, shown_img1)

            shown_img2 = cv2.drawKeypoints(img2, self.list2.kp, shown_img2, color=(0, 255, 0), flags=0)
            cv2.imwrite(kp_path2, shown_img2)
        else:
            if self.list1.bbox.top_left_x != 0 or self.list1.bbox.top_left_y != 0:
                for idx in range(self.list1.length):
                    cv2.circle(shown_img1, (int(self.list1.pt_x[idx] + self.list1.bbox.top_left_x),
                                            int(self.list1.pt_y[idx] + self.list1.bbox.top_left_y)),
                               3, color=(0, 255, 0))
                cv2.imwrite(kp_path1, shown_img1)

            if self.list2.bbox.top_left_x != 0 or self.list2.bbox.top_left_y != 0:
                for idx in range(self.list2.length):
                    cv2.circle(shown_img2, (int(self.list2.pt_x[idx] + self.list2.bbox.top_left_x),
                                            int(self.list2.pt_y[idx] + self.list2.bbox.top_left_y)),
                               3, color=(0, 255, 0))
                cv2.imwrite(kp_path2, shown_img2)
