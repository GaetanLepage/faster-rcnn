import numpy as np
import tensorflow as tf

from faster_rcnn.utils.anchor import Anchor



def create_anchor_objects(image_width,
                          image_height,
                          feature_map_width,
                          feature_map_height,
                          anchors_area_list,
                          anchors_aspect_ratio_list):
    """
    TODO
    """

    anchor_list = []

    # We iterate over all sliding window positions (i.e. every pixel of the feature map)
    for x_anchor_center_in_fm in range(feature_map_width):
        for y_anchor_center_in_fm in range(feature_map_height):

            # Transpose anchor center coordinates from feature map to image coordinates
            x_anchor_center_in_image = int(image_width / feature_map_width) * x_anchor_center_in_fm
            y_anchor_center_in_image = int(image_height / feature_map_height) * y_anchor_center_in_fm

            for anchor_area in anchors_area_list:
                for anchor_aspect_ratio in anchors_aspect_ratio_list:


                    anchor = Anchor(area=anchor_area,
                                    aspect_ratio=anchor_aspect_ratio,
                                    x_center_in_image=x_anchor_center_in_image,
                                    y_center_in_image=y_anchor_center_in_image,
                                    x_center_in_feature_map=x_anchor_center_in_fm,
                                    y_center_in_feature_map=y_anchor_center_in_fm,
                                    ground_truth_bounding_box=None)

                    # During training, we ignore anchor boxes that cross image boundaries
                    if not anchor.is_crossing_image_boundaries(image_width=image_width,
                                                               image_height=image_height):
                        anchor_list.append(anchor)

    return anchor_list



def map_anchors_and_gt_bbox(ground_truth_bbox_list,
                            anchor_list):
    """
    TODO
    """

    for ground_truth_bbox in ground_truth_bbox_list:

        max_iou = 0
        best_matching_anchors = []

        for anchor in anchor_list:

            iou = anchor.iou_with_box(ground_truth_bbox)

            # Record, for each anchor, the IoU overlap with the best matching bounding box
            if iou > anchor.highest_iou_with_ground_truth_bbox:
                anchor.highest_iou_with_ground_truth_bbox = iou

            # Condition (ii) from Faster-RCNN paper (3.1.2)
            # The anchor is positive
            if iou > 0.7:
                anchor.assign_bounding_box(ground_truth_bbox)


            # Record the anchors having the hiighest IoU overlap with the current bounding box
            if iou == max_iou:
                best_matching_anchors.append(anchor)

            elif iou > max_iou:
                max_iou = iou

                best_matching_anchors = [anchor]


        # Condition (i) from Faster-RCNN paper (3.1.2)
        # The anchors with the highest IoU with the current bounding box are positive
        for anchor in best_matching_anchors:
            anchor.assign_bounding_box(ground_truth_bbox)

    # Marking negative anchors:
    # i.e. the layers having an IoU < 0.3 for all ground truth boxes
    for anchor in anchor_list:

        if anchor.highest_iou_with_ground_truth_bbox < 0.3:
            anchor.label = False

    # Remove all the anchors which are neither positive nor negative
    anchor_list = list(filter(lambda anchor: anchor.label is not None,
                              iterable=anchor_list))

    return anchor_list


def sample(anchor_list,
           num_anchors):
    """
    TODO

    Args:
        anchor_list: TODO
        num_anchors: the expected number of anchors after the sampling procedure

    Returns:
        TODO
    """
    # First, we shuffle the anchor list
    np.random.shuffle(anchor_list)

    positive_anchors = list(filter(lambda anchor: anchor.label,
                                   iterable=anchor_list))
    num_positive = min(len(positive_anchors), num_anchors / 2)
    positive_anchors = positive_anchors[:num_positive]

    num_negative = num_anchors - num_positive

    negative_anchors = list(filter(lambda anchor: not anchor.label,
                                   iterable=anchor_list))[:num_negative]

    sampled_anchors = positive_anchors + negative_anchors

    assert len(sampled_anchors) == num_anchors, "Bad number of sampled anchors"

    return sampled_anchors



def generate_gt_tensors(anchor_list,
                        anchors_aspect_ratio_list,
                        anchors_area_list,
                        feature_map_width,
                        feature_map_height):
    """
    TODO
    """
    num_anchors = len(anchors_area_list) * len(anchors_aspect_ratio_list)

    mask = np.zeros(shape=(feature_map_height, feature_map_width, num_anchors))

    cls_shape = (feature_map_height, feature_map_width, num_anchors)
    y_true_cls = np.zeros(shape=cls_shape)

    reg_shape = (feature_map_height, feature_map_width, 4 * num_anchors)
    y_true_reg = np.zeros(shape=reg_shape)

    for anchor in anchor_list:
        # First dimension index [0, feature_map_width]
        x_index = anchor.x_center_in_feature_map
        # Second dimension index [0, feature_map_height]
        y_index = anchor.y_center_in_feature_map
        # Third dimension index [0, num_anchors]
        aspect_ratio_index = anchors_aspect_ratio_list.index(anchor.aspect_ratio)
        area_index = anchors_area_list.index(anchor.area_in_pixels)
        anchor_index = area_index * len(anchors_area_list) + aspect_ratio_index

        ## Mask
        mask[y_index][x_index][anchor_index] = 1


        ## Classification

        y_true_cls[y_index][x_index][anchor_index] = int(anchor.label)
        # Index of the first coordinate
        # prob_start_index = 2 * base_prob_index
        # prob_index_mask = [prob_start_index, prob_start_index + 1]

        # y_true_cls[y_index][x_index][pred_box_start_index] = int(anchor.label)
        # y_true_cls[y_index][x_index][pred_box_start_index + 1] = 1 - int(anchor.label)


        ## Regression

        # Index of the first coordinate
        pred_box_start_index = 4 * anchor_index
        pred_box_index_mask = range(pred_box_start_index, pred_box_start_index + 4)

        y_true_reg[y_index][x_index][pred_box_index_mask] = anchor.reg_vector


    # Convert numpy arrays to tensors
    mask = tf.convert_to_tensor(value=mask,
                                dtype=tf.bool)

    y_true_cls = tf.convert_to_tensor(value=y_true_cls,
                                      dtype=tf.float32)

    y_true_reg = tf.convert_to_tensor(value=y_true_reg,
                                      dtype=tf.float32)


    return mask, y_true_cls, y_true_reg



def pre_process_img(image_width,
                    image_height,
                    feature_map_width,
                    feature_map_height,
                    ground_truth_bbox_list,
                    anchors_area_list,
                    anchors_aspect_ratio_list,
                    num_anchors):
    """
    TODO
    """


    anchor_list = create_anchor_objects(image_width,
                                        image_height,
                                        feature_map_width,
                                        feature_map_height,
                                        anchors_area_list,
                                        anchors_aspect_ratio_list)



    anchor_list = map_anchors_and_gt_bbox(ground_truth_bbox_list=ground_truth_bbox_list,
                                          anchor_list=anchor_list)

    # Select only num_anchors anchor boxes and preserve (as much as possible)
    # balance between classes
    anchor_list = sample(anchor_list,
                         num_anchors=num_anchors)

    # Generate ground truth tensors that have same shapes as the network output layers
    ground_truth_tensors = generate_gt_tensors(anchor_list=anchor_list,
                                               anchors_aspect_ratio_list=anchors_aspect_ratio_list,
                                               anchors_area_list=anchors_area_list,
                                               feature_map_width=feature_map_width,
                                               feature_map_height=feature_map_height)


    return ground_truth_tensors
