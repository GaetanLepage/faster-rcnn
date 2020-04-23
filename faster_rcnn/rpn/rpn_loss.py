import tensorflow as tf
from faster_rcnn.utils.box import get_center


@tf.function
def _robust_loss(y_true, y_pred):
    """
    TODO
    """

    diff = y_pred - y_true

    # |x| < 1
    condition = tf.less_equal(tf.abs(diff), 1)

    value_1 = 0.5 * (diff ** 2)
    value_2 = tf.abs(diff) - tf.constant(value=0.5, shape=tf.shape(diff))

    result = value_1 * condition + value_2 * (1 - condition)

    return result


# class RpnLoss:

    # def __init__(self,
                 # anchors_aspect_ratio_list,
                 # anchors_area_list,
                 # n_cls=None,
                 # n_reg=None,
                 # lambda_balance=10):
        # """
        # TODO

        # Args:
            # n_cls: mini-batch size
            # n_reg: the number number of anchor locations
            # lambda_balance: the balancing parameter
        # """

        # self.anchors_aspect_ratio_list = anchors_aspect_ratio_list
        # self.anchors_area_list = anchors_area_list
        # self.n_cls = n_cls
        # self.n_reg = n_reg
        # self.lambda_balance = lambda_balance

        # self.cls_predictions = tf.Variable()

def rpn_loss_wrapper(lambda_balance=10):
    """
    TODO

    Args:
        n_cls: mini-batch size
        n_reg: the number number of anchor locations
        lambda_balance: the balancing parameter
    """

    @tf.function
    def rpn_loss(ground_truth_tensors, y_pred):
        """
        TODO
        """

        n_cls = tf.shape(ground_truth_tensors)[3]

        feature_map_height, feature_map_width = tf.shape(ground_truth_tensors)[1:2]
        n_reg = feature_map_height * feature_map_width

        mask, y_true_cls, y_true_reg = ground_truth_tensors

        cls_predictions, reg_predictions = y_pred


        ## Classification

        # Compute the binary classification crossentropy loss
        cls_loss_tensor = tf.keras.backend.binary_crossentropy(target=y_true_cls,
                                                               output=cls_predictions)

        # Keep only the loss of the selected anchors
        cls_loss_tensor = tf.boolean_mask(tensor=cls_loss_tensor,
                                          mask=mask)

        # Sum all the classification losses
        cls_loss = tf.reduce_sum(cls_loss_tensor)




        ## Regression
        reg_loss_tensor = _robust_loss(y_true=y_true_reg,
                                       y_pred=reg_predictions)

        cls_gt_mask = tf.cast(x=y_true_cls,
                              dtype=tf.bool)
        # from (B, H, W, num_anchors) to (B, H, W, 4*num_anchors)
        cls_gt_mask = tf.keras.backend.repeat_elements(x=cls_gt_mask,
                                                       rep=4,
                                                       axis=-1)
        # The regression loss is only computed for anchors which
        #   contain an object.
        reg_loss_tensor = tf.boolean_mask(tensor=reg_loss_tensor,
                                          mask=cls_gt_mask)


        # from (B, H, W, num_anchors) to (B, H, W, 4*num_anchors)
        reg_mask = tf.keras.backend.repeat_elements(x=mask,
                                                    rep=4,
                                                    axis=-1)
        # Regression loss is only computed for selected anchors
        reg_loss_tensor = tf.boolean_mask(tensor=reg_loss_tensor,
                                          mask=reg_mask)

        # Sum all the regression losses
        reg_loss = tf.reduce_sum(reg_loss_tensor)


        return (1 / n_cls) * cls_loss + (lambda_balance / n_reg) * reg_loss

    return rpn_loss
