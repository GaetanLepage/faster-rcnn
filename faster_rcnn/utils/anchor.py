"""
TODO
"""

import numpy as np
import math as m
from faster_rcnn.utils import box


class Anchor:


    def __init__(self,
                 area_in_pixels,
                 aspect_ratio,
                 x_center_in_image,
                 y_center_in_image,
                 x_center_in_feature_map,
                 y_center_in_feature_map,
                 ground_truth_bounding_box):

        self.area_in_pixels = area_in_pixels
        self.aspect_ratio = aspect_ratio

        # Either 'True' (positive anchor), 'False' (negative anchor) or None
        self.label = None

        self.x_center_in_image = x_center_in_image
        self.y_center_in_image = y_center_in_image

        self.x_center_in_feature_map = x_center_in_feature_map
        self.y_center_in_feature_map = y_center_in_feature_map

        self.coordinates = self.get_anchor_coordinates()

        self.ground_truth_bounding_box = ground_truth_bounding_box
        self.highest_iou_with_ground_truth_bbox = 0


    def get_anchor_coordinates(self):
        """
        TODO
        """

        self.width = m.sqrt(self.area_in_pixels * self.aspect_ratio)
        self.height = m.sqrt(self.area_in_pixels / self.aspect_ratio)

        x_min_anchor = self.x_center_in_image - (self.width / 2)
        y_min_anchor = self.y_center_in_image - (self.height / 2)
        x_max_anchor = self.x_center_in_image + (self.width / 2)
        y_max_anchor = self.y_center_in_image + (self.height / 2)

        anchor_coordinates = [x_min_anchor,
                              y_min_anchor,
                              x_max_anchor,
                              y_max_anchor]

        return anchor_coordinates


    def is_crossing_image_boundaries(self, image_width, image_height):
        """
        TODO
        """

        if self.coordinates[0] < 0:
            return True

        if self.coordinates[1] < 0:
            return True

        if self.coordinates[2] > image_width:
            return True

        if self.coordinates[3] > image_height:
            return True

        return False



    def iou_with_box(self, bounding_box):
        """
        TODO
        """

        return box.iou(rectangle_1=self.get_anchor_coordinates(),
                       rectangle_2=bounding_box)


    def assign_bounding_box(self, bounding_box):
        """
        TODO
        """

        # Mark the anchor as positive
        self.label = True

        iou_with_new_box = self.iou_with_box(bounding_box)

        # Assign this bounding box to this anchor if this ground truth box is closer
        if iou_with_new_box > self.highest_iou_with_ground_truth_bbox:

            self.ground_truth_bounding_box_coordinates = bounding_box
            self.highest_iou_with_ground_truth_bbox = iou_with_new_box

            # Refresh the reg vector
            gt_bbox_x, gt_bbox_y = box.get_center(bounding_box)
            t_x = (gt_bbox_x - self.x_center_in_image) / self.width
            t_y = (gt_bbox_y - self.y_center_in_image) / self.height

            gt_bbox_width, gt_bbox_height = box.get_width_height(bounding_box)
            t_w = np.log(gt_bbox_width / self.width)
            t_h = np.log(gt_bbox_height / self.height)

            self.reg_vector = (t_x, t_y, t_w, t_h)
