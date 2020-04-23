"""
TODO
"""



import tensorflow as tf
import tensorflow.keras.backend as K



class RoiMaxPooling(tf.keras.layers.Layer):
    """
    TODO
    """

    def __init__(self,
                 output_feature_map_shape,
                 name='roi_max_pooling',
                 **kwargs):
        """
        TODO
        """
        super(RoiMaxPooling, self).__init__(name=name, **kwargs)

        self.grid_height, self.grid_width = output_feature_map_shape


    def build(self, input_shape):
        """
        TODO
        """
        self.batch_size = input_shape[0]
        self.nb_channels = input_shape[3]


    def call(self, inputs):
        """
        TODO
        """
        input_shape = tf.shape(inputs)

        #h/H
        bin_height = input_shape[1] / self.grid_height
        #w/W
        bin_width = input_shape[2] / self.grid_width
        pool_size = (bin_height, bin_width)

        outputs = []

        for h_bin_index in range(self.grid_width):
            for v_bin_index in range(self.grid_height):

                x1 = h_bin_index * bin_width
                x2 = x1 + bin_width
                y1 = v_bin_index * bin_height
                y2 = y1 + bin_height

                x1 = tf.cast(x=tf.round(x1),
                             dtype=tf.int32)
                x2 = tf.cast(x=tf.round(x2),
                             dtype=tf.int32)
                y1 = tf.cast(x=tf.round(y1),
                             dtype=tf.int32)
                y2 = tf.cast(x=tf.round(y2),
                             dtype=tf.int32)

                cropped_input = inputs[:, y1:y2, x1:x2, :]

                new_shape = [self.batch_size,
                             y2 - y1,
                             x2 - x1,
                             self.nb_channels]
                cropped_input = tf.reshape(tensor=cropped_input,
                                           shape=new_shape)

                pooled_values = tf.keras.backend.max(x=cropped_input,
                                                     axis=(1, 2))

                outputs.append(pooled_values)

        final_output = tf.concat(values=outputs,
                                 axis=0)

        output_shape = (self.batch_size,
                        self.grid_height,
                        self.grid_width,
                        self.nb_channels)

        final_output = tf.reshape(tensor=final_output,
                                  shape=output_shape)

        return final_output



def get_fast_rcnn_model(num_categories,
                        grid_height,
                        grid_width):
    """
    TODO
    """

    # The input of the netowrk is batch of regions of interest (RoI)
    # of shape (B, h, w, c) where c is the number of channels
    inputs = tf.keras.Input(shape=(None, None, 3),
                            name='fast_rcnn_input_layer')


    # (B, h, w, c) --> (B, H, W, c)
    feature_bins = RoiMaxPooling(output_feature_map_shape=(grid_height, grid_width),
                                 name='toi_max_pooling')

    flattened_feature_bins = tf.keras.layers.Flatten(name="flatten")(feature_bins)

    # Common fully connected layers
    feature_vector = tf.keras.layers.Dense(units=64)(flattened_feature_bins)
    feature_vector = tf.keras.layers.Dense(units=32)(feature_vector)

    # Classification fully connected layer
    cls_output = tf.keras.layers.Dense(units=num_categories + 1,
                                       name='fast_rcnn_cls')(feature_vector)

    # Regression fully connected layer
    reg_output = tf.keras.layers.Dense(units=4 * num_categories,
                                       name='fast_rcnn_reg')(feature_vector)


    return tf.keras.Model(inputs=inputs,
                          outputs=[cls_output, reg_output])
