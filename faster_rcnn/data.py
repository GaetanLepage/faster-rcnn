"""
TODO
"""

import tensorflow as tf
import tensorflow_datasets as tfds

from faster_rcnn.rpn.pre_processing import pre_process_img


def normalize(input_image):
    """
    Normalize the given image and cast it to a float32 tensor.

    Args:
        input_image: An image tensor with shape (H, W, c) of values between
                        0 and 254.

    Returns:
        A tensor of float32 values between 0 and 1.
    """

    return tf.cast(x=input_image,
                   dtype=tf.float32) / 255.0

def load_image_train(input_image,
                     objects,
                     feature_map_shape,
                     anchors_area_list,
                     anchors_aspect_ratio_list):
    """
    TODO
    """

    # print("image : ", input_image)
    # print("objects : ", objects)
    # print("feature_map_shape : ", feature_map_shape)
    # print("anchors_area_list : ", anchors_area_list)
    # print("anchors_aspect_ratio_list : ", anchors_aspect_ratio_list)

    input_image_shape = tf.shape(input=input_image)

    # TODO maybe do some random data transformation here

    gt_bbox_list = tf.unstack(objects)

    anchors_area_list = tf.cast(x=anchors_area_list,
                                dtype=tf.float32)

    ground_truth_tensors = pre_process_img(image_shape=input_image_shape,
                                           feature_map_shape=feature_map_shape,
                                           ground_truth_bbox_list=gt_bbox_list,
                                           anchors_area_list=anchors_area_list,
                                           anchors_aspect_ratio_list=anchors_aspect_ratio_list)

    result = normalize(input_image), *ground_truth_tensors

    return result


def get_load_image_function(feature_map_shape,
                            anchors_area_list,
                            anchors_aspect_ratio_list):
    """
    TODO
    """
    def load_image_train(input_image,
                         objects):
        """
        TODO
        """

        # input_image, objects = datapoint

        input_image_shape = tf.shape(input=input_image)

        # TODO maybe do some random data transformation here

        gt_bbox_list = tf.unstack(objects)

        # gt_bbox_list = [object_element for object_element in objects['bbox']]

        ground_truth_tensors = pre_process_img(image_shape=input_image_shape,
                                               feature_map_shape=feature_map_shape,
                                               ground_truth_bbox_list=gt_bbox_list,
                                               anchors_area_list=anchors_area_list,
                                               anchors_aspect_ratio_list=anchors_aspect_ratio_list)

        return normalize(input_image), ground_truth_tensors

    return load_image_train
