import tensorflow as tf


class Module(object):
    """
    RNN Module which gets as an input the belief of predicates and objects
    and outputs an improved belief for predicates and objects
    """

    def __init__(self, nof_predicates, nof_objects, visual_features_predicate_size, visual_features_object_size,
                 rnn_steps=1, is_train=True):
        """
        Construct module:
        - create input placeholders
        - create rnn step
        - attach rnn_step rnn_steps times
        - create labels placeholders
        - create module loss and train_step

        :param nof_predicates: nof predicate labels
        :param nof_objects: nof object labels
        :param visual_features_predicate_size: predicate visual features size
        :param visual_features_object_size: object visual features size
        :param rnn_steps: rnn length
        :param is_train: whether the module will be used to train or eval
        """
        ## save input
        self.nof_predicates = nof_predicates
        self.nof_objects = nof_objects
        self.visual_features_predicate_size = visual_features_predicate_size
        self.visual_features_object_size = visual_features_object_size
        self.is_train = is_train

        ## module input
        # Visual features
        self.visual_features_predicate_ph = tf.placeholder(dtype=tf.float32,
                                                           shape=(None, None, self.visual_features_predicate_size),
                                                           name="visual_feautres_predicate")
        self.visual_features_object_ph = tf.placeholder(dtype=tf.float32,
                                                        shape=(None, self.visual_features_object_size),
                                                        name="visual_features_object")
        # belief
        self.belief_predicate_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, self.nof_predicates),
                                                  name="belief_predicate")
        self.belief_object_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.nof_objects), name="belief_object")

        ## labels
        self.labels_predicate_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, self.nof_predicates),
                                                  name="labels_predicate")
        self.labels_object_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.nof_objects), name="labels_object")

        # single rnn stage module
        self.out_belief_predicate, self.out_belief_object = \
            self.rnn_stage(in_visual_features_predicate=self.visual_features_predicate_ph,
                           in_visual_features_object=self.visual_features_object_ph,
                           in_belief_predicate=self.belief_predicate_ph,
                           in_belief_object=self.belief_object_ph)

        # loss
        self.loss, self.train_step = self.module_loss()

    def nn(self, features, out_size, out_shape, scope_name="nn"):
        """
        simple nn to convert features to belief
        :param features: features tensor
        :param out_size: nof labels of a belief
        :param out_shape: output shape (used to reshape to required output shape)
        :param scope_name: tensorflow scope name
        :return:
        """
        in_size = features.shape[-1]._value
        #h1_size = 2 * in_size
        h1_size = 100
        #h2_size = 2 * in_size
        h2_size = 100

        with tf.variable_scope(scope_name):
            # Define the initialization of the first layer
            w_1 = tf.get_variable(name="w1", shape=(in_size, h1_size),
                                  initializer=tf.truncated_normal_initializer())
            b_1 = tf.get_variable(name="b1", shape=(h1_size),
                                  initializer=tf.truncated_normal_initializer())

            # Define the initialization of the second layer
            w_2 = tf.get_variable(name="w2", shape=(h1_size, h2_size),
                                  initializer=tf.truncated_normal_initializer())
            b_2 = tf.get_variable(name="b2", shape=(h2_size),
                                  initializer=tf.truncated_normal_initializer())

            # Define the initialization of the third layer
            w_3 = tf.get_variable(name="w3", shape=(h2_size, out_size),
                                  initializer=tf.truncated_normal_initializer())
            b_3 = tf.get_variable(name="b3", shape=(out_size),
                                  initializer=tf.truncated_normal_initializer())

            # Create neural network
            input_features = tf.reshape(features, (-1, in_size))
            h1 = tf.nn.tanh(tf.matmul(input_features, w_1) + b_1, name="h1")
            h2 = tf.nn.tanh(tf.matmul(h1, w_2) + b_2, name="h2")
            y = tf.add(tf.matmul(h2, w_3), b_3, name="y")
            if not self.is_train:
                y = tf.nn.softmax(y, name="y")

            # reshape to fit the required output dims
            out = tf.reshape(y, out_shape)

        return out

    def rnn_stage(self, in_visual_features_predicate, in_visual_features_object, in_belief_predicate, in_belief_object,
                  scope_name="rnn_cell"):
        """
        RNN stage - which get as an input a belief of the predicates and objects and return an improved belief of the predicates and the objects
        :param in_visual_features_predicate:
        :param in_visual_features_object:
        :param in_belief_predicate:
        :param in_belief_object:
        :param socpe_name:
        :return:
        """
        with tf.variable_scope(scope_name):
            # get global subject belief
            global_sub_belief = tf.reduce_max(in_belief_predicate, axis=1, name="global_sub_belief")
            # expand global sub belief
            expand_global_sub_belief = tf.add(tf.zeros_like(in_belief_predicate), global_sub_belief,
                                              name="expand_global_sub_belief")

            # get global object belief
            global_obj_belief = tf.reduce_max(in_belief_predicate, axis=0, name="global_obj_belief")
            # expand global sub belief
            expand_global_obj_belief = tf.add(tf.zeros_like(in_belief_predicate), global_obj_belief,
                                              name="expand_global_obj_belief")

            # expand visual object features
            expand_visual_object = tf.add(tf.zeros_like(in_visual_features_predicate), in_visual_features_object,
                                          name="expand_visual_object")

            # expand visual subbject features
            expand_visual_subject = tf.transpose(expand_visual_object, perm=[1, 0, 2], name="expand_visual_subject")

            predicate_all_features = tf.concat(
                (in_visual_features_predicate, in_belief_predicate, expand_global_sub_belief, expand_global_obj_belief,
                 expand_visual_subject, expand_visual_object),
                axis=2, name="predicate_all_features")

            # object all features
            object_all_features = tf.concat(
                (in_visual_features_object, global_sub_belief, global_obj_belief, in_belief_object),
                axis=1, name="object_all_features")

            # fully cnn to calc belief predicate for every subject and object
            out_belief_predicate = self.nn(predicate_all_features, out_size=self.nof_predicates,
                                      out_shape=tf.shape(in_belief_predicate), scope_name="predicate")

            # fully cnn to calc belief object for every object
            out_belief_object = self.nn(object_all_features, out_size=self.nof_objects, out_shape=tf.shape(in_belief_object),
                                   scope_name="object")

            return out_belief_predicate, out_belief_object

    def module_loss(self, lr=0.1, lr_steps=1000, lr_decay=0.5, scope_name="loss"):
        """
        Set and minimize module loss
        :param lr: init learning rate
        :param lr_steps: steps to decay learning rate
        :param lr_decay: factor to decay the learning rate by
        :param scope_name: tensor flow scope name
        :return: loss and train step
        """
        with tf.variable_scope(scope_name):
            # reshape to batch like shape
            shaped_belief_predicate = tf.reshape(self.out_belief_predicate, (-1, self.nof_predicates))
            shaped_labels_predicate = tf.reshape(self.labels_predicate_ph, (-1, self.nof_predicates))

            # set loss
            loss_predicate = tf.nn.softmax_cross_entropy_with_logits(labels=shaped_labels_predicate,
                                                                     logits=shaped_belief_predicate,
                                                                     name="loss_predicate")
            loss_object = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_object_ph, logits=self.out_belief_object,
                                                                  name="loss_object")
            loss = tf.add(tf.reduce_sum(loss_predicate), tf.reduce_sum(loss_object), name="loss")

            # minimize
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(lr, global_step, lr_steps,
                                                       lr_decay, staircase=True)
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        return loss, train_step

    def get_in_ph(self):
        """
        get input place holders
        """
        return self.belief_predicate_ph, self.belief_object_ph, self.visual_features_predicate_ph, self.visual_features_object_ph

    def get_output(self):
        """
        get module output
        """
        return self.out_belief_predicate, self.out_belief_object

    def get_labels_ph(self):
        """
        get module labels ph (used for train)
        """
        return self.labels_predicate_ph, self.labels_object_ph

    def get_module_loss(self):
        """
        get module loss and train step
        """
        return self.loss, self.train_step
