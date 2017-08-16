import tensorflow as tf


class Module(object):
    """
    RNN Module which gets as an input the belief of predicates and objects
    and outputs an improved belief for predicates and objects
    """
    # FIXME: use single rnn step at first
    def __init__(self, nof_predicates, nof_objects, visual_features_predicate_size, visual_features_object_size,
                 rnn_steps=1, is_train=True,
                 learning_rate=0.1, learning_rate_steps=1000, learning_rate_decay=0.5):
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
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_steps = learning_rate_steps
        self.learning_rate = learning_rate
        self.nof_predicates = nof_predicates
        self.nof_objects = nof_objects
        self.visual_features_predicate_size = visual_features_predicate_size
        self.visual_features_object_size = visual_features_object_size
        self.is_train = is_train
        self.rnn_steps = rnn_steps

        ## create weights
        # FIXME: input features size change to include just predicate + subject + object belief
        # self.nn_predicate_weights(self.nof_predicates * 3 + self.visual_features_predicate_size + 2 * self.nof_objects, self.nof_predicates)
        self.nn_predicate_weights(self.nof_predicates + 2 * self.nof_objects, self.nof_predicates)
        # FIXME: don't create weights for object nn 
        #self.nn_object_weights(self.nof_predicates * 2 + self.nof_objects + self.visual_features_object_size, self.nof_objects)

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

        self.extended_belief_object_shape_ph = tf.placeholder(dtype=tf.int32, shape=(3), name="extended_belief_object_shape")

        # labels
        if self.is_train:
            self.labels_predicate_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, self.nof_predicates),
                                                      name="labels_predicate")
            self.labels_object_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.nof_objects), name="labels_object")
            self.labels_coeff_loss_ph = tf.placeholder(dtype=tf.float32, shape=(None), name="labels_coeff_loss")
        # single rnn stage module
        belief_predicate = self.belief_predicate_ph
        belief_object = self.belief_object_ph

        for step in range(self.rnn_steps):
            # FIXME: don't modify object belief
            belief_predicate, _, last_layer_predicate, last_layer_object = \
            self.rnn_stage(in_visual_features_predicate=self.visual_features_predicate_ph,
                           in_visual_features_object=self.visual_features_object_ph,
                           in_belief_predicate=belief_predicate,
                           in_belief_object=belief_object,
                           in_extended_belief_object_shape=self.extended_belief_object_shape_ph,
                           scope_name="rnn" + str(step))

        self.out_belief_predicate = belief_predicate
        self.out_belief_object = belief_object
        self.last_layer_predicate = last_layer_predicate
        self.last_layer_object = last_layer_object

        # loss
        if self.is_train:
            # Learning rate
            self.lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name="lr_ph")
        
            self.loss, self.gradients, self.grad_placeholder, self.train_step = self.module_loss()
            
            

    def nn_predicate_weights(self, in_size, out_size):
        # FIXME: larger hidden layers for testing
        h1_size = 1000
        h2_size = 1000
        h3_size = 1000
        h4_size = 1000
        with tf.variable_scope("nn_predicate_weights"):
            # create predicate nn weights just once for all rnn stages
            # Define the initialization of the first layer
            self.nn_predicate_w_1 = tf.get_variable(name="w1", shape=(in_size, h1_size),
                                                    initializer=tf.truncated_normal_initializer())
            self.nn_predicate_b_1 = tf.get_variable(name="b1", shape=(h1_size),
                                                    initializer=tf.truncated_normal_initializer())

            # Define the initialization of the second layer
            self.nn_predicate_w_2 = tf.get_variable(name="w2", shape=(h1_size, h2_size),
                                                    initializer=tf.truncated_normal_initializer())
            self.nn_predicate_b_2 = tf.get_variable(name="b2", shape=(h2_size),
                                                    initializer=tf.truncated_normal_initializer())

            # Define the initialization of the third layer
            self.nn_predicate_w_3 = tf.get_variable(name="w3", shape=(h2_size, h3_size),
                                                    initializer=tf.truncated_normal_initializer())
            self.nn_predicate_b_3 = tf.get_variable(name="b3", shape=(h3_size),
                                                    initializer=tf.truncated_normal_initializer())
            # Define the initialization of the layer 4
            self.nn_predicate_w_4 = tf.get_variable(name="w4", shape=(h3_size, h4_size),
                                                    initializer=tf.truncated_normal_initializer())
            self.nn_predicate_b_4 = tf.get_variable(name="b4", shape=(h4_size),
                                                    initializer=tf.truncated_normal_initializer())
            # Define the initialization of the layer 5
            self.nn_predicate_w_5 = tf.get_variable(name="w5", shape=(h4_size, out_size),
                                                    initializer=tf.truncated_normal_initializer())
            self.nn_predicate_b_5 = tf.get_variable(name="b5", shape=(out_size),
                                                    initializer=tf.truncated_normal_initializer())


    def nn_predicate(self, features, in_belief_predicate, out_shape, scope_name="nn_predicate"):
        """
        simple nn to convert features to belief
        :param features: features tensor
        :param out_shape: output shape (used to reshape to required output shape)
        :param scope_name: tensorflow scope name
        :return:
        """
        in_size = features.shape[-1]._value
        with tf.variable_scope(scope_name):
            # Create neural network
            input_features = tf.reshape(features, (-1, in_size))
            
            h1 = tf.nn.tanh(tf.matmul(input_features, self.nn_predicate_w_1) + self.nn_predicate_b_1, name="h1")
            h2 = tf.nn.tanh(tf.matmul(h1, self.nn_predicate_w_2) + self.nn_predicate_b_2, name="h2")
            h3 = tf.nn.tanh(tf.matmul(h2, self.nn_predicate_w_3) + self.nn_predicate_b_3, name="h3")
            h4 = tf.nn.tanh(tf.matmul(h3, self.nn_predicate_w_4) + self.nn_predicate_b_4, name="h4")
            delta = tf.nn.tanh(tf.add(tf.matmul(h4, self.nn_predicate_w_5), self.nn_predicate_b_5, name="delta"))
            in_belief_shaped = tf.reshape(in_belief_predicate, tf.shape(delta))
            y = tf.add(delta, in_belief_shaped, name="y")
            #y = delta

            out = tf.nn.softmax(y, name="out")

            # reshape to fit the required output dims
            y = tf.reshape(y, out_shape)
            out = tf.reshape(out, out_shape)

        return out , y
    def nn_object_weights(self, in_size, out_size):
        h1_size = 200
        h2_size = 200
        h3_size = 200
        h4_size = 200

        with tf.variable_scope("nn_object_weights"):
            # Define the initialization of the first layer
            self.nn_object_w_1 = tf.get_variable(name="w1", shape=(in_size, h1_size),
                                                 initializer=tf.truncated_normal_initializer())
            self.nn_object_b_1 = tf.get_variable(name="b1", shape=(h1_size),
                                                 initializer=tf.truncated_normal_initializer())

            # Define the initialization of the second layer
            self.nn_object_w_2 = tf.get_variable(name="w2", shape=(h1_size, h2_size),
                                                 initializer=tf.truncated_normal_initializer())
            self.nn_object_b_2 = tf.get_variable(name="b2", shape=(h2_size),
                                                 initializer=tf.truncated_normal_initializer())

            # Define the initialization of the layer 3
            self.nn_object_w_3 = tf.get_variable(name="w3", shape=(h2_size, h3_size),
                                                 initializer=tf.truncated_normal_initializer())
            self.nn_object_b_3 = tf.get_variable(name="b3", shape=(h3_size),
                                                 initializer=tf.truncated_normal_initializer())
            # Define the initialization of the layer 4
            self.nn_object_w_4 = tf.get_variable(name="w4", shape=(h3_size, h4_size),
                                                 initializer=tf.truncated_normal_initializer())
            self.nn_object_b_4 = tf.get_variable(name="b4", shape=(h4_size),
                                                 initializer=tf.truncated_normal_initializer())
            # Define the initialization of the layer 5
            self.nn_object_w_5 = tf.get_variable(name="w5", shape=(h4_size, out_size),
                                                 initializer=tf.truncated_normal_initializer())
            self.nn_object_b_5 = tf.get_variable(name="b5", shape=(out_size),
                                                 initializer=tf.truncated_normal_initializer())

    def nn_object(self, features, in_belief_object, out_size, scope_name="nn_object"):
        """
        simple nn to convert features to belief
        :param features: features tensor
        :param out_size: nof labels of a belief
        :param scope_name: tensorflow scope name
        :return:
        """

        with tf.variable_scope(scope_name):

            # Create neural network
            h1 = tf.nn.tanh(tf.matmul(features, self.nn_object_w_1) + self.nn_object_b_1, name="h1")
            h2 = tf.nn.tanh(tf.matmul(h1, self.nn_object_w_2) + self.nn_object_b_2, name="h2")
            h3 = tf.nn.tanh(tf.matmul(h2, self.nn_object_w_3) + self.nn_object_b_3, name="h3")
            h4 = tf.nn.tanh(tf.matmul(h3, self.nn_object_w_4) + self.nn_object_b_4, name="h4")
            delta = tf.nn.tanh(tf.add(tf.matmul(h4, self.nn_object_w_5), self.nn_object_b_5, name="delta"))
            y = tf.add(delta, in_belief_object, name="y")

            out = tf.nn.softmax(y, name="out")

        return out, y

    def rnn_stage(self, in_visual_features_predicate, in_visual_features_object, in_belief_predicate, in_belief_object,
                  in_extended_belief_object_shape, scope_name="rnn_cell"):
        """
        RNN stage - which get as an input a belief of the predicates and objects and return an improved belief of the predicates and the objects
        :return:
        :param in_visual_features_predicate: visual features of predicate
        :param in_visual_features_object: visual features of the object
        :param in_belief_predicate: predicate belief of the last stage in the RNN
        :param in_belief_object: object belief of the last stage in the RNNS
        :param in_extended_belief_object_shape: the shape of the extended version of object belief (N, N, NOF_OBJECTS)
        :param scope_name: rnn stage scope
        :return:
        """
        with tf.variable_scope(scope_name):
            with tf.variable_scope("feature_collector"):
                # get global subject belief
                global_sub_belief = tf.reduce_max(in_belief_predicate, axis=1, name="global_sub_belief")
                self.global_sub_belief = global_sub_belief
                # expand global sub belief
                expand_global_sub_belief = tf.add(tf.zeros_like(in_belief_predicate), global_sub_belief,
                                                  name="expand_global_sub_belief")
                self.expand_global_sub_belief = expand_global_sub_belief
                # get global object belief
                global_obj_belief = tf.reduce_max(in_belief_predicate, axis=0, name="global_obj_belief")
                self.global_obj_belief = global_obj_belief
                # expand global sub belief
                expand_global_obj_belief = tf.add(tf.zeros_like(in_belief_predicate), global_obj_belief,
                                                  name="expand_global_obj_belief")
                self.expand_global_obj_belief = expand_global_obj_belief
                # expand visual object features
                expand_belief_object = tf.add(tf.zeros(in_extended_belief_object_shape), in_belief_object,
                                              name="expand_belief_object")
                self.expand_belief_object = expand_belief_object
                # expand visual subject features
                expand_belief_subject = tf.transpose(expand_belief_object, perm=[1, 0, 2], name="expand_belief_subject")
                self.expand_belief_subject = expand_belief_subject
                
                # FIXME: concat just the beliefs for a simpler network
                predicate_all_features = tf.concat(
                    (in_belief_predicate, expand_belief_subject, expand_belief_object),
                    axis=2, name="predicate_all_features")
                #    (in_visual_features_predicate, in_belief_predicate, expand_global_sub_belief, expand_global_obj_belief,

                # object all features
                object_all_features = tf.concat(
                    (in_visual_features_object, global_sub_belief, global_obj_belief, in_belief_object),
                    axis=1, name="object_all_features")

            # fully cnn to calc belief predicate for every subject and object
            out_belief_predicate, last_layer_predicate = self.nn_predicate(predicate_all_features, in_belief_predicate,
                                      out_shape=tf.shape(in_belief_predicate))

            # fully cnn to calc belief object for every object
            # FIXME: don't run object nn for now
            #out_belief_object, last_layer_object = self.nn_object(object_all_features, in_belief_object, out_size=self.nof_objects)
            out_belief_object = in_belief_object
            last_layer_object = in_belief_object
            return out_belief_predicate, out_belief_object, last_layer_predicate, last_layer_object

    def module_loss(self, scope_name="loss"):
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
            shaped_belief_predicate = tf.reshape(self.last_layer_predicate, (-1, self.nof_predicates))
            shaped_labels_predicate = tf.reshape(self.labels_predicate_ph, (-1, self.nof_predicates))
            # set loss
            self.loss_predicate = tf.nn.softmax_cross_entropy_with_logits(labels=shaped_labels_predicate,
                                                                     logits=shaped_belief_predicate,
                                                                     name="loss_predicate")
            self.loss_predicate_weighted = tf.multiply(self.loss_predicate, self.labels_coeff_loss_ph)
            # FIXME: don't calc object loss
            #loss_object = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_object_ph, logits=self.last_layer_object,
            #                                                      name="loss_object")
            #loss = tf.add(tf.reduce_mean(self.loss_predicate_weighted), 2 * tf.reduce_mean(loss_object), name="loss")
            #loss = tf.reduce_mean(loss_object)
            # FIXME: take into account just predicate loss
            loss = tf.reduce_mean(self.loss_predicate_weighted)

            # minimize
            gradients = tf.train.GradientDescentOptimizer(self.lr_ph).compute_gradients(loss)
            # create placeholder to minimize in a batch
            grad_placeholder = [(tf.placeholder("float", shape=grad[0].get_shape()), grad[1]) for grad in gradients]
            train_step = tf.train.GradientDescentOptimizer(self.lr_ph).apply_gradients(grad_placeholder)
        return loss, gradients, grad_placeholder, train_step

    def get_in_ph(self):
        """
        get input place holders
        """
        return self.belief_predicate_ph, self.belief_object_ph, self.extended_belief_object_shape_ph, self.visual_features_predicate_ph, self.visual_features_object_ph

    def get_output(self):
        """
        get module output
        """
        return self.out_belief_predicate, self.out_belief_object

    def get_labels_ph(self):
        """
        get module labels ph (used for train)
        """
        return self.labels_predicate_ph, self.labels_object_ph, self.labels_coeff_loss_ph

    def get_module_loss(self):
        """
        get module loss and train step
        """
        return self.loss, self.gradients, self.grad_placeholder, self.train_step
