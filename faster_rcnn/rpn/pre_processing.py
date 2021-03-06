import numpy as np
import tensorflow as tf

from faster_rcnn.utils.anchor import Anchor



def _create_anchor_objects(image_shape,
                           feature_map_shape,
                           anchors_area_list,
                           anchors_aspect_ratio_list):
    """
    TODO
    """

    anchor_list = []

    feature_map_height = feature_map_shape[0]
    feature_map_width = feature_map_shape[1]

    # We iterate over all sliding window positions (i.e. every pixel of the feature map)
    for x_anchor_center_in_fm in range(feature_map_width):
        for y_anchor_center_in_fm in range(feature_map_height):

            # Transpose anchor center coordinates to normalized coordinates
            x_anchor_center_normalized = float(x_anchor_center_in_fm) / float(feature_map_width)
            y_anchor_center_normalized = float(y_anchor_center_in_fm) / float(feature_map_height)

            for anchor_area in anchors_area_list:
                for anchor_aspect_ratio in anchors_aspect_ratio_list:


                    anchor = Anchor(area=anchor_area,
                                    aspect_ratio=anchor_aspect_ratio,
                                    x_center=x_anchor_center_normalized,
                                    y_center=y_anchor_center_normalized,
                                    image_shape=image_shape,
                                    ground_truth_bounding_box=None)

                    # During training, we ignore anchor boxes that cross image boundaries
                    if not anchor.is_crossing_image_boundaries():
                        anchor_list.append(anchor)

    return anchor_list



def _map_anchors_and_gt_bbox(ground_truth_bbox_list,
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
                              anchor_list))

    return anchor_list


def _sample(anchor_list,
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
                                   anchor_list))
    num_positive = min(len(positive_anchors), int(num_anchors / 2))

    positive_anchors = positive_anchors[:num_positive]

    num_negative = num_anchors - num_positive

    negative_anchors = list(filter(lambda anchor: not anchor.label,
                                   anchor_list))[:num_negative]

    sampled_anchors = positive_anchors + negative_anchors

    assert len(sampled_anchors) == num_anchors, "Bad number of sampled anchors"

    return sampled_anchors



def _generate_gt_tensors(anchor_list,
                         anchors_aspect_ratio_list,
                         anchors_area_list,
                         feature_map_shape):
    """
    TODO
    """
    num_anchors = len(anchors_area_list) * len(anchors_aspect_ratio_list)

    feature_map_height = feature_map_shape[0]
    feature_map_width = feature_map_shape[1]

    mask = np.zeros(shape=(feature_map_height, feature_map_width, num_anchors))

    cls_shape = (feature_map_height, feature_map_width, num_anchors)
    y_true_cls = np.zeros(shape=cls_shape)

    reg_shape = (feature_map_height, feature_map_width, 4 * num_anchors)
    y_true_reg = np.zeros(shape=reg_shape)

    for anchor in anchor_list:
        # First dimension index [0, feature_map_width]
        x_index = int(anchor.x_center * float(feature_map_width))
        # Second dimension index [0, feature_map_height]
        y_index = int(anchor.y_center * float(feature_map_height))
        # Third dimension index [0, num_anchors]
        aspect_ratio_index = np.array(anchors_aspect_ratio_list).tolist().index(anchor.aspect_ratio)
        area_index = np.array(anchors_area_list).tolist().index(anchor.area)
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
                                dtype=tf.bool,
                                name='mask')

    y_true_cls = tf.convert_to_tensor(value=y_true_cls,
                                      dtype=tf.bool,
                                      name='y_true_cls')

    y_true_reg = tf.convert_to_tensor(value=y_true_reg,
                                      dtype=tf.float32,
                                      name='y_true_reg')


    return mask, y_true_cls, y_true_reg



def pre_process_img(image_shape,
                    feature_map_shape,
                    ground_truth_bbox_list,
                    anchors_area_list,
                    anchors_aspect_ratio_list):
    """
    TODO
    """
    import time

    print("start preprocessing")

    start = time.time()

    anchor_list = _create_anchor_objects(image_shape,
                                         feature_map_shape,
                                         anchors_area_list,
                                         anchors_aspect_ratio_list)

    create_anchor_time = time.time() - start
    print("create_anchor_time done :", create_anchor_time)


    anchor_list = _map_anchors_and_gt_bbox(ground_truth_bbox_list=ground_truth_bbox_list,
                                           anchor_list=anchor_list)

    map_time = time.time() - create_anchor_time - start
    print("map anchors and gt bbox :", map_time)

    # Select only num_anchors anchor boxes and preserve (as much as possible)
    # balance between classes
    anchor_list = _sample(anchor_list,
                          num_anchors=len(anchors_area_list) * len(anchors_aspect_ratio_list))

    sampling_time = time.time() - map_time -start
    print("sample :", sampling_time)

    # Generate ground truth tensors that have same shapes as the network output layers
    ground_truth_tensors = _generate_gt_tensors(anchor_list=anchor_list,
                                                anchors_aspect_ratio_list=anchors_aspect_ratio_list,
                                                anchors_area_list=anchors_area_list,
                                                feature_map_shape=feature_map_shape)

    tensor_time = time.time() - sampling_time - start
    print("generated gt tensors :", tensor_time)

    print("TOTAL TIME =", time.time() - start)


    return ground_truth_tensors
