import os
import timeit
import numpy as np

from bbox.boundingbox import BoundingBox
from keypoint_detection import keypoint_pairs

from keypoint_detection.ORB import ORB_point
from keypoint_detection.keypoint_pairs import KeypointPairList
from keypoint_matching.brute_force import BruteForceMatcherList

__author__ = 'Dandi Chen'


img_path = '/mnt/scratch/haoyiliang/KITTI/SceneFLow2015/data_scene_flow/training/image_2/'
flow_gt_path = '/mnt/scratch/haoyiliang/KITTI/SceneFLow2015/data_scene_flow/training/flow_occ'

eval_path = '/mnt/scratch/DandiChen/keypoint/KITTI/optimization/confidence/'
kp_path = eval_path + 'keypoint/'
match_path = eval_path + 'matches/000000_10_customized/box/weighted_dis/w-0.5/'


img_num = len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))])
flow_num = img_num - 1  # continuous two frames

# pair_num = img_num/2
# pair_num = 2
pair_num = 1

t = []

for img in range(pair_num):
    print ''
    print 'img number: ', img

    img_path1 = os.path.join(img_path, str(img).zfill(6) + '_10.png')
    img_path2 = os.path.join(img_path, str(img).zfill(6) + '_11.png')

    # start = timeit.default_timer()
    img1, img2 = keypoint_pairs.read_img_pair(img_path1, img_path2)
    height, width, _ = img1.shape

    start = timeit.default_timer()

    # bounding box coordinates
    bbox1 = BoundingBox(154.07749939, 181.342102051, 405.574401855, 305.924407959)
    bbox2 = BoundingBox(0.0, 156.604873657, 353.453063965, 351.0)

    # ORB keypoint
    orb1 = ORB_point(bbox1)
    orb2 = ORB_point(bbox2)
    orb1.get_keypoint(img1[int(bbox1.top_left_y):int(bbox1.bottom_right_y),
                      int(bbox1.top_left_x):int(bbox1.bottom_right_x)])
    orb2.get_keypoint(img2[int(bbox2.top_left_y):int(bbox2.bottom_right_y),
                      int(bbox2.top_left_x):int(bbox2.bottom_right_x)])
    t_kp = timeit.default_timer()
    print 'keypoint extraction time:', t_kp - start
    print 'keypoint number:', orb1.length, orb2.length
    print ''
    orb_pair = KeypointPairList(orb1, orb2)
    # orb_pair.vis_pt_pairs(img1, img2)
    # orb_pair.write_pt_pairs(img1, img2, os.path.join(kp_path, str(img).zfill(6) + '_10.png'),
    #                         os.path.join(kp_path, str(img).zfill(6) + '_11.png'))

    # BFMatcher
    t_matcher_0 = timeit.default_timer()
    bfm = BruteForceMatcherList(orb_pair)
    bfm.get_matcher()
    t_matcher = timeit.default_timer()
    print 'matcher number:', bfm.length
    print 'matcher time :', t_matcher - t_matcher_0
    print ''

    t_good_matcher_0 = timeit.default_timer()
    bfm.get_good_matcher()           # Lowe's good feature threshold criteria(feature similarity distance)
    t_good_matcher = timeit.default_timer()
    print 'good matcher number:', bfm.length
    print 'good matcher time:', t_good_matcher - t_good_matcher_0
    print ''

    # find homography
    t_wgt_matcher_0 = timeit.default_timer()
    bfm.get_wgt_dis_matcher()
    t_wgt_matcher = timeit.default_timer()
    print 'good weighted matcher number:', bfm.length
    print 'good weighted matcher time:', t_wgt_matcher - t_wgt_matcher_0
    print ''

    # find homography
    t_homography_0 = timeit.default_timer()
    bfm.get_homography()
    t_homography = timeit.default_timer()
    print 'homography time:', t_good_matcher - t_good_matcher_0
    print ''

    # bfm.vis_matches(img1, img2, 1, 0, bfm.length)
    # bfm.write_matches(img1, img2, os.path.join(match_path, str(img).zfill(6) + '_10_non_overlap_match.png'),
    #                   1, 0, bfm.length)
    # bfm.write_matches_overlap(img1, img2, os.path.join(match_path, str(img).zfill(6) + '_10_overlap_match.png'),
    #                           1, 0, bfm.length)

    end = timeit.default_timer()
    print 'total time = ', end - start
    t.append([end - start])

print ''
print 'ave time = ', np.mean(t)
