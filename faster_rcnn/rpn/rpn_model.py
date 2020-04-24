"""
TODO
"""


import tensorflow as tf

def _get_base_model():

    base_model = tf.keras.applications.MobileNetV2(weights='imagenet',
                                                   include_top=False)

    # Feature map layer
    output_layer = base_model.get_layer('block_6_expand_relu')

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input,
                                outputs=output_layer.output,
                                name='feature_extractor')

    down_stack.trainable = False

    # down_stack.summary()

    # tf.keras.utils.plot_model(down_stack,
                              # to_file="model.png",
                              # show_shapes=True)

    return down_stack

@tf.function
def _convert_reg_output(pred_vector,
                       anchor_width,
                       anchor_height,
                       x_anchor_center,
                       y_anchor_center):

    t_y, t_x, t_h, t_w = pred_vector

    y_pred_bbox_center = t_y * anchor_height + y_anchor_center
    x_pred_bbox_center = t_x * anchor_width + x_anchor_center

    h_pred_bbox = anchor_height * tf.exp(t_h)
    w_pred_bbox = anchor_width * tf.exp(t_w)

    return [y_pred_bbox_center,
            x_pred_bbox_center,
            h_pred_bbox,
            w_pred_bbox]


@tf.function
def rpn_post_processing(cls_output,
                        reg_output,
                        anchors_aspect_ratio_list,
                        anchors_area_list,
                        batch_image_shapes,
                        ):
    """
    TODO
    """

    reg_shape = tf.shape(input=reg_output)

    batch_size = reg_shape[0]
    feature_map_height = reg_shape[1]
    feature_map_width = reg_shape[2]
    num_anchors = reg_shape[3] / 4

    for image_index in range(batch_size):

        image_shape = batch_image_shapes[image_index]
        image_height = image_shape[0]
        image_width = image_shape[1]
        image_reg_output = reg_output[image_index]

        assert tf.shape(image_reg_output) == image_shape,\
                "The provided image shape is invalid."

        for y_index in range(feature_map_height):

            for x_index in range(feature_map_width):

                for anchor_index in range(num_anchors):

                    pred_vector = image_reg_output[y_index][y_index][anchor_index:anchor_index + 4]

                    anchor_area = anchors_area_list[
                        anchor_index // len(anchors_area_list)]
                    anchor_aspect_ratio = anchors_aspect_ratio_list[
                        anchor_index % len(anchors_area_list)]

                    anchor_width = tf.sqrt(anchor_area * anchor_aspect_ratio)
                    anchor_height = tf.sqrt(anchor_area / anchor_aspect_ratio)

                    # Transpose anchor center coordinates from feature map to image coordinates
                    x_anchor_center_in_image = int(image_width / feature_map_width) * x_index
                    y_anchor_center_in_image = int(image_height / feature_map_height) * y_index


                    bbox_pred_coordinates = _convert_reg_output(pred_vector,
                                                               anchor_width,
                                                               anchor_height,
                                                               x_anchor_center=x_anchor_center_in_image,
                                                               y_anchor_center=y_anchor_center_in_image)



    # manage bbox that extend past the image boundaries

    # Perform NMS





def get_rpn_model(sliding_window_size, num_anchors):
    """
    RPN model builder

    TODO remove : Builds the global Faster-RCNN model

    Args:
        sliding_window_size:    the size of the sliding window (n in the paper)
        num_windows:            the number of anchors

    Returns:
        a tf.keras.Model including the feature extractor and the object detection patches
    """

    inputs = tf.keras.Input(shape=(None, None, 3),
                            name='rpn_input_layer')

    feature_extractor = _get_base_model()

    feature_map = feature_extractor(inputs)

    # n_channels = feature_extractor.output_shape[-1]
    n_channels = feature_map.shape[-1]

    sliding_windows_features = tf.keras.layers.Conv2D(filters=n_channels,
                                                      kernel_size=sliding_window_size,
                                                      activation='relu',
                                                      name='rpn_intermediate_layer')(feature_map)

    cls_output = tf.keras.layers.Conv2D(filters=num_anchors,
                                        kernel_size=1,
                                        activation='sigmoid',
                                        name='rpn_cls')(sliding_windows_features)


    reg_output = tf.keras.layers.Conv2D(filters=4 * num_anchors,
                                        kernel_size=1,
                                        name='rpn_reg')(sliding_windows_features)

    return tf.keras.Model(inputs=inputs,
                          outputs=[cls_output, reg_output])
