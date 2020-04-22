import tensorflow as tf
from faster_rcnn.utils.box import get_center



def _classification_loss(true_label, predicted_prob):
    """
    TODO
    """

    return

def _regression_loss(anchor, predicted_bbox):
    """
    TODO
    """

    ground_truth_bounding_box = anchor.ground_truth_bounding_box
    x_gt_bbox_center, y_gt_bbox_center = get_center(ground_truth_bounding_box)

    x_predicted_bbox_center, y_predicted_bbox_center = get_center(predicted_bbox)

    x_anchor_center = anchor.x_center_in_image
    y_anchor_center = anchor.y_center_in_image

    t_x = (x_predicted_bbox_center - x_anchor_center) / anchor.width
    t_y = (y_predicted_bbox_center - y_anchor_center) / anchor.height

    w_pred_box = tf.math.abs()

    t_w = tf.math.log()

    ground_truth_t_vector = None
    predicted_t_vector = None


def rpn_loss_wrapper(anchors_aspect_ratio_list,
                     anchors_area_list,
                     n_cls=None,
                     n_reg=None,
                     lambda_balance=10):
    """
    TODO

    Args:
        n_cls: mini-batch size
        n_reg: the number number of anchor locations
        lambda_balance: the balancing parameter
    """

    def rpn_loss(anchor_list, network_predictions):
        """
        TODO
        """

        class_predictions, regression_predictions = network_predictions[0], network_predictions[1]

        cls_loss = 0
        reg_loss = 0

        for anchor in anchor_list:
            true_label = int(anchor.label)

            aspect_ratio_index = anchors_aspect_ratio_list.index(anchor.aspect_ratio)
            area_index = anchors_area_list.index(anchor.area_in_pixels)
            x_index = anchor.x_center_in_feature_map
            y_index = anchor.y_center_in_feature_map
            prob_index = 2 * (area_index * len(anchors_area_list) + aspect_ratio_index)

            # Get the predicted probability from the cls output layer
            # shape=[FM_WIDTH, FM_HEIGHT, 2*k]
            # where :
            #   * FM_WIDTH is the width of the feature map
            #   * FM_HEIGHT is the height of the feature map
            #   * k is the number of anchor boxes per location
            #       k = |anchors_aspect_ratio_list| * |anchors_area_list|
            predicted_prob = class_predictions[x_index][y_index][prob_index]


            cls_loss += _classification_loss(true_label, predicted_prob)


            # Index of the first coordinate
            pred_box_start_index = 2 * (area_index * len(anchors_area_list) + aspect_ratio_index)

            # Gather the four coordinates of the predicted bounding box
            pred_box_index_mask = range(pred_box_start_index, pred_box_start_index + 4)
            predicted_coordinates = regression_predictions[x_index][y_index][pred_box_index_mask]

            reg_loss += _regression_loss(anchor=anchor,
                                         predicted_bbox=predicted_coordinates)

        total_loss = (1 / n_cls) * cls_loss + (lambda_balance / n_reg) * reg_loss

    return rpn_loss




