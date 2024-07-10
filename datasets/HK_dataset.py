from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
from .mono_dataset import MonoDataset


class HKInitDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, flipping = False, rotating = False, distorted=False, inpaint_pseudo_gt_dir = None, **kwargs):
        super(HKInitDataset, self).__init__(*args, **kwargs)

        self.inpaint_pseudo_gt_dir = inpaint_pseudo_gt_dir
        self.flipping = flipping
        self.rotating = rotating
        
        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        # HK from MeanIntrinsics.ipynb for C3VD data: 
        if distorted:
            intrinsics = [0.7428879629629629, 0.7424861111111111, 0.4937833333333333, 0.5071601851851851]
        else:
            intrinsics = [0.6423119966210151, 0.6401273085242651, 0.4824200466376491, 0.5298353978680292]
        
        # should i assume 0.5 0.5 center????? no i will remove flip
        # also add c3vd option
        self.K = np.array([[intrinsics[0], 0, intrinsics[2], 0],
                           [0, intrinsics[1], intrinsics[3], 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (288, 288)

    def check_depth(self):
        
        return False
    
    def get_color(self, folder, frame_index, side, do_flip, do_rot):
        color = self.loader(self.get_image_path(folder, frame_index, side)) #pil image

        if do_flip and self.flipping:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        if do_rot and self.rotating:
            angle = np.random.choice([pil.ROTATE_90, pil.ROTATE_180, pil.ROTATE_270])
            color = color.transpose(angle)
        return color
    

class HKDataset(HKInitDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(HKDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:05d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(folder, f_str)
        return image_path