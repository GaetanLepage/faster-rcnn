"""
TODO
"""

import numpy as np
import math as m
from faster_rcnn.utils import box

import tensorflow as tf


class Anchor:
    """
    Class representing an anchor.

    Attributes:
        area: The area of the box in squared pixels
        aspect_ratio: The aspect ration of the anchor box.
        x_center_in_image, y_center_in_image: coordinates of the center of the anchor box
                                                 in the original image setting.
        x_center_in_feature_map, y_center_in_feature_map: coordinates of the center of the anchor
                                                            box in the feature map setting.
        highest_iou_with_ground_truth_bbox: Contains the value of the IoU overlap with the closest
                                                ground truth bounding box yet came across.
        label: Classification ground truth (whether the anchor 'contains' an object)
                   * 'True' : anchor is positive
                   * 'False': anchor is negative
                   * 'None' : anchor is neither positive nor negative
        ground_truth_bounding_box: The coordinates [x1, y1, x2, y2] of the associated ground truth
                                    bounding box.
    """


    def __init__(self,
                 area,
                 aspect_ratio,
                 x_center,
                 y_center,
                 image_shape,
                 ground_truth_bounding_box):

        self.area = area
        self.aspect_ratio = aspect_ratio

        # Either 'True' (positive anchor), 'False' (negative anchor) or None
        self.label = None
        self.reg_vector = None

        self.image_shape = image_shape

        self.x_center = x_center
        self.y_center = y_center

        self.coordinates = self.get_anchor_coordinates()

        self.ground_truth_bounding_box = ground_truth_bounding_box
        self.highest_iou_with_ground_truth_bbox = 0


    def get_anchor_coordinates(self):
        """
        Computes the coordinates of the anchor box in the original image setting.

        Returns:
            [y1, x1, y2, x2] the coordinates of the anchor box.
        """

        width_in_pixel = m.sqrt(self.area * self.aspect_ratio)
        height_in_pixel = m.sqrt(self.area / self.aspect_ratio)

        image_height, image_width, _ = self.image_shape

        self.width = width_in_pixel / float(image_width)
        self.height = height_in_pixel / float(image_height)

        # print("anchor area =", self.area)
        # print("anchor aspect_ratio =", self.aspect_ratio)
        # print("anchor width  in pixels = ", width_in_pixel)
        # print("image width = ", image_width)
        # print("=> anchor normalized width = ", self.width)

        # print("anchor height in pixels = ", height_in_pixel)
        # print("image height = ", image_height)
        # print("=> anchor normalized height = ", self.height)
        # print("###########################")

        x_min_anchor = self.x_center - self.width / 2
        y_min_anchor = self.y_center - self.height / 2
        x_max_anchor = self.x_center + self.width / 2
        y_max_anchor = self.y_center + self.height / 2

        anchor_coordinates = tf.convert_to_tensor(value=[y_min_anchor,
                                                         x_min_anchor,
                                                         y_max_anchor,
                                                         x_max_anchor],
                                                  dtype=tf.float32)

        return anchor_coordinates


    def is_crossing_image_boundaries(self):
        """
        Check whether the anchor crosses the image boundary.

        Args:
            image_width, image_height: shape of the image (in pixels)

        Returns:
            True if the anchor is cross-boundary.
            False otherwise.
        """

        y_min, x_min, y_max, x_max = self.coordinates

        if y_min < 0:
            return True

        if x_min < 0:
            return True

        if y_max > 1:
            return True

        if x_max > 1:
            return True

        return False



    def iou_with_box(self, bounding_box):
        """
        Compute the Intersection over Union of the anchor box and any other bounding box.

        Args:
            bounding_box: [y1, x1, y2, x2] some bounding box.

        Returns:
            iou (float): the value of the IoU.
        """

        return box.iou(rectangle_1=self.get_anchor_coordinates(),
                       rectangle_2=bounding_box)


    def assign_bounding_box(self, bounding_box):
        """
        If the given bounding box has a better IoU than the currently assigned
        ground truth bounding box, then it is assigned to this anchor box.

        The coordinates in the 'regression parametrization' [tx, ty, tw, th] are also computed.

        Args:
            bounding_box: [y1, x1, y2, x2] some bounding box.
        """

        # Mark the anchor as positive
        self.label = True

        # Compute the IoU with the given box
        iou_with_new_box = self.iou_with_box(bounding_box)

        # Assign this bounding box to this anchor if this ground truth box is closer
        if iou_with_new_box > self.highest_iou_with_ground_truth_bbox:

            self.ground_truth_bounding_box = bounding_box
            self.highest_iou_with_ground_truth_bbox = iou_with_new_box

            # Refresh the reg vector
            gt_bbox_x, gt_bbox_y = box.get_center(bounding_box)
            t_x = (gt_bbox_x - self.x_center) / self.width
            t_y = (gt_bbox_y - self.y_center) / self.height

            gt_bbox_width, gt_bbox_height = box.get_width_height(bounding_box)
            t_w = np.log(gt_bbox_width / self.width)
            t_h = np.log(gt_bbox_height / self.height)

            self.reg_vector = (t_x, t_y, t_w, t_h)
