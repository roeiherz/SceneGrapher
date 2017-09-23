import tensorflow as tf


class Module(object):
    """
    RNN Module which gets as an input the belief of predicates and objects
    and outputs an improved belief for predicates and objects
    """
    def __init__(self, nof_predicates, nof_objects, visual_features_predicate_size, visual_features_object_size,
                 rnn_steps=2, is_train=True, loss_func="all",
                 learning_rate=0.1, learning_rate_steps=1000, learning_rate_decay=0.5,
                 including_object=False, lr_object_coeff=1):
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
        self.nof_features = self.nof_predicates * 3 + 3 * self.nof_objects
        self.loss_func = loss_func
        self.including_object = including_object
        self.lr_object_coeff = lr_object_coeff

        ## create weights
        self.nn_predicate_weights(self.nof_features, self.nof_predicates)
        if including_object:
            self.nn_object_weights(self.nof_predicates * 2 + self.nof_objects * 2, self.nof_objects)

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


        # shape to be used by feature collector
        self.extended_belief_object_shape_ph = tf.placeholder(dtype=tf.int32, shape=(3), name="extended_belief_object_shape")

        # labels
        if self.is_train:
            self.labels_predicate_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, self.nof_predicates),
                                                      name="labels_predicate")
            self.labels_object_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.nof_objects), name="labels_object")
            self.labels_coeff_loss_ph = tf.placeholder(dtype=tf.float32, shape=(None), name="labels_coeff_loss")
        
        # store all the outputs of of rnn steps
        self.out_belief_object_lst = []
        self.out_belief_predicate_lst = []
        # rnn stage module
        belief_predicate = self.belief_predicate_ph
        belief_object = self.belief_object_ph

        for step in range(self.rnn_steps):

            belief_predicate, belief_object_temp = \
            self.rnn_stage(in_belief_predicate=belief_predicate,
                           in_belief_object=belief_object,
                           in_extended_belief_object_shape=self.extended_belief_object_shape_ph,
                           scope_name="rnn" + str(step))
            # store the belief
            self.out_belief_predicate_lst.append(belief_predicate)
            if self.including_object:
                belief_object = belief_object_temp
                # store the belief
                self.out_belief_object_lst.append(belief_object_temp)

        self.out_belief_predicate = belief_predicate
        self.out_belief_object = belief_object
        reshaped_predicate_belief = tf.reshape(belief_predicate, (-1, self.nof_predicates))
        reshaped_predicete_probes = tf.nn.softmax(reshaped_predicate_belief)
        self.out_predicate_probes = tf.reshape(reshaped_predicete_probes, tf.shape(belief_predicate), name="out_predicate_probes")
        self.out_object_probes = tf.nn.softmax(belief_object, name="out_object_probes")

        # loss
        if self.is_train:
            # Learning rate
            self.lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name="lr_ph")
        
            self.loss, self.gradients, self.grad_placeholder, self.train_step = self.module_loss()
            
            

    def nn_predicate_weights(self, in_size, out_size):
        h1_size = 500
        h2_size = 500
        h3_size = 500
        h4_size = 500
        with tf.variable_scope("nn_predicate_weights"):
            # create predicate nn weights just once for all rnn stages
            # Define the initialization of the first layer
            self.nn_predicate_w_1 = tf.get_variable(name="w1", shape=(in_size, h1_size),
                                                    initializer=tf.truncated_normal_initializer(stddev=0.03))
            self.nn_predicate_b_1 = tf.get_variable(name="b1", shape=(h1_size),
                                                    initializer=tf.truncated_normal_initializer(stddev=0.03))

            # Define the initialization of the second layer
            self.nn_predicate_w_2 = tf.get_variable(name="w2", shape=(h1_size, h2_size),
                                                    initializer=tf.truncated_normal_initializer(stddev=0.03))
            self.nn_predicate_b_2 = tf.get_variable(name="b2", shape=(h2_size),
                                                    initializer=tf.truncated_normal_initializer(stddev=0.03))

            # Define the initialization of the third layer
            self.nn_predicate_w_3 = tf.get_variable(name="w3", shape=(h2_size, h3_size),
                                                    initializer=tf.truncated_normal_initializer(stddev=0.03))
            self.nn_predicate_b_3 = tf.get_variable(name="b3", shape=(h3_size),
                                                    initializer=tf.truncated_normal_initializer(stddev=0.03))
            # Define the initialization of the layer 4
            self.nn_predicate_w_4 = tf.get_variable(name="w4", shape=(h3_size, h4_size),
                                                    initializer=tf.truncated_normal_initializer(stddev=0.03))
            self.nn_predicate_b_4 = tf.get_variable(name="b4", shape=(h4_size),
                                                    initializer=tf.truncated_normal_initializer(stddev=0.03))
            # Define the initialization of the layer 5
            self.nn_predicate_w_5 = tf.get_variable(name="w5", shape=(h4_size	, out_size),
                                                    initializer=tf.truncated_normal_initializer(stddev=0.03))
            self.nn_predicate_b_5 = tf.get_variable(name="b5", shape=(out_size),
                                                    initializer=tf.truncated_normal_initializer(stddev=0.03))


    def nn_predicate(self, features, in_belief_predicate, out_shape, scope_name="nn_predicate"):
        """
        simple nn to convert features to belief
        :param features: features tensor
        :param out_shape: output shape (used to reshape to required output shape)
        :param scope_name: tensorflow scope name
        :return: predicate probes and predicate belief
        """
        in_size = features.shape[-1]._value
        with tf.variable_scope(scope_name):
            # Create neural network
            input_features = tf.reshape(features, (-1, in_size))
            #if self.is_train:
            #     input_features = tf.nn.dropout(input_features, self.keep_prob_ph)

            h1 = tf.nn.relu(tf.matmul(input_features, self.nn_predicate_w_1) + self.nn_predicate_b_1, name="h1")
            h2 = tf.nn.relu(tf.matmul(h1, self.nn_predicate_w_2) + self.nn_predicate_b_2, name="h2")
            h3 = tf.nn.relu(tf.matmul(h2, self.nn_predicate_w_3) + self.nn_predicate_b_3, name="h3")
            h4 = tf.nn.relu(tf.matmul(h3, self.nn_predicate_w_4) + self.nn_predicate_b_4, name="h4")
            h5 = tf.add(tf.matmul(h4, self.nn_predicate_w_5), self.nn_predicate_b_5, name="y")
            self.predicate_delta = h5
            in_belief_shaped = tf.reshape(in_belief_predicate, tf.shape(h5))
            y = tf.add(h5, in_belief_shaped, name="y")

            # reshape to fit the required output dims
            y = tf.reshape(y, out_shape)

        return y

    def nn_object_weights(self, in_size, out_size):
        h1_size = 500
        h2_size = 500
        h3_size = 500
        h4_size = 500

        with tf.variable_scope("nn_object_weights"):
            # Define the initialization of the first layer
            self.nn_object_w_1 = tf.get_variable(name="w1", shape=(in_size, h1_size),
                                                 initializer=tf.truncated_normal_initializer(stddev=0.03))
            self.nn_object_b_1 = tf.get_variable(name="b1", shape=(h1_size),
                                                 initializer=tf.truncated_normal_initializer(stddev=0.03))

            # Define the initialization of the second layer
            self.nn_object_w_2 = tf.get_variable(name="w2", shape=(h1_size, h2_size),
                                                 initializer=tf.truncated_normal_initializer(stddev=0.03))
            self.nn_object_b_2 = tf.get_variable(name="b2", shape=(h2_size),
                                                 initializer=tf.truncated_normal_initializer(stddev=0.03))

            # Define the initialization of the layer 3
            self.nn_object_w_3 = tf.get_variable(name="w3", shape=(h2_size, h3_size),
                                                 initializer=tf.truncated_normal_initializer(stddev=0.03))
            self.nn_object_b_3 = tf.get_variable(name="b3", shape=(h3_size),
                                                 initializer=tf.truncated_normal_initializer(stddev=0.03))
            # Define the initialization of the layer 4
            self.nn_object_w_4 = tf.get_variable(name="w4", shape=(h3_size, h4_size),
                                                 initializer=tf.truncated_normal_initializer(stddev=0.03))
            self.nn_object_b_4 = tf.get_variable(name="b4", shape=(h4_size),
                                                 initializer=tf.truncated_normal_initializer(stddev=0.03))
            # Define the initialization of the layer 5
            self.nn_object_w_5 = tf.get_variable(name="w5", shape=(h4_size, out_size),
                                                 initializer=tf.truncated_normal_initializer(stddev=0.03))
            self.nn_object_b_5 = tf.get_variable(name="b5", shape=(out_size),
                                                 initializer=tf.truncated_normal_initializer(stddev=0.03))

    def nn_object(self, features, in_belief_object, scope_name="nn_object"):
        """
        simple nn to convert features to belief
        :param features: features tensor
        :param scope_name: tensorflow scope name
        :return: object probabilities and object belief
        """

        with tf.variable_scope(scope_name):

            # Create neural network
            h1 = tf.nn.relu(tf.matmul(features, self.nn_object_w_1) + self.nn_object_b_1, name="h1")
            h2 = tf.nn.relu(tf.matmul(h1, self.nn_object_w_2) + self.nn_object_b_2, name="h2")
            h3 = tf.nn.relu(tf.matmul(h2, self.nn_object_w_3) + self.nn_object_b_3, name="h3")
            h4 = tf.nn.relu(tf.matmul(h3, self.nn_object_w_4) + self.nn_object_b_4, name="h4")
            self.object_delta = tf.add(tf.matmul(h4, self.nn_object_w_5), self.nn_object_b_5, name="delta")
            y = tf.add(self.object_delta, in_belief_object, name="y")


        return y

    def rnn_stage(self, in_belief_predicate, in_belief_object,
                  in_extended_belief_object_shape, scope_name="rnn_cell"):
        """
        RNN stage - which get as an input a belief of the predicates and objects and return an improved belief of the predicates and the objects
        :return:
        :param in_belief_predicate: predicate belief of the last stage in the RNN
        :param in_belief_object: object belief of the last stage in the RNNS
        :param in_extended_belief_object_shape: the shape of the extended version of object belief (N, N, NOF_OBJECTS)
        :param scope_name: rnn stage scope
        :return: improved predicates probabilties, improved predicate belief,  improved object probabilites and improved object belief
        """
        with tf.variable_scope(scope_name):
            with tf.variable_scope("feature_collector"):
                # mean center
                in_belief_predicate = in_belief_predicate - tf.reduce_mean(in_belief_predicate, axis=2, keep_dims=True)
                in_belief_object = in_belief_object - tf.reduce_mean(in_belief_object, axis=1, keep_dims=True)

                # get likelihood to be included in sg for specific subject
                all_subjects_predicates = tf.reduce_max(in_belief_predicate, axis=1, name="all_subjects_predicates")
                # expand to NxN
                expand_all_subject_predicates = tf.add(tf.zeros_like(in_belief_predicate), all_subjects_predicates)
                expand_all_subject_predicates = tf.transpose(expand_all_subject_predicates, perm=[1, 0, 2], name="expand_all_subject_predicates")

                # get likelihood to be included in sg for specific subject
                all_object_predicates = tf.reduce_max(in_belief_predicate, axis=0, name="all_object_predicates")
                # expand to NxN
                expand_global_obj_belief = tf.add(tf.zeros_like(in_belief_predicate), all_object_predicates,
                                                  name="expand_global_obj_belief")

                # expand object belief
                expand_belief_object = tf.add(tf.zeros(in_extended_belief_object_shape), in_belief_object,
                                              name="expand_belief_object")

                # expand subject belief
                expand_belief_subject = tf.transpose(expand_belief_object, perm=[1, 0, 2], name="expand_belief_subject")
                self.expand_belief_subject = expand_belief_subject

                # expand object belief
                all_object_belief = tf.reduce_max(in_belief_object, axis=0, name="all_object_belief")
                expand_all_object_belief_2d = tf.add(tf.zeros_like(in_belief_object), all_object_belief, name="expand_all_object_belief_2d")
                expand_all_object_belief_3d = tf.add(tf.zeros(in_extended_belief_object_shape), all_object_belief, name="expand_all_object_belief_3d")


                predicate_all_features = tf.concat(
                    (in_belief_predicate, expand_belief_subject, expand_belief_object, expand_all_subject_predicates, expand_global_obj_belief, expand_all_object_belief_3d), axis=2, name="predicate_all_features")

                # object all features
                object_all_features = tf.concat(
                    (in_belief_object, all_subjects_predicates, all_object_predicates, expand_all_object_belief_2d),
                    axis=1, name="object_all_features")

            # fully cnn to calc belief predicate for every subject and object
            out_belief_predicate = self.nn_predicate(predicate_all_features, in_belief_predicate,
                                                                       out_shape=tf.shape(in_belief_predicate))

            # fully cnn to calc belief object for every object
            if self.including_object:
                out_belief_object = self.nn_object(object_all_features, in_belief_object)
            else:
                out_belief_object = in_belief_object



            return out_belief_predicate, out_belief_object

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
            shaped_labels_predicate = tf.reshape(self.labels_predicate_ph, (-1, self.nof_predicates))
            shaped_in_belief_predicate = tf.reshape(self.belief_predicate_ph, (-1, self.nof_predicates))
            
            loss = 0
            for rnn_step in range(self.rnn_steps):
            
            
                shaped_belief_predicate = tf.reshape(self.out_belief_predicate_lst[rnn_step], (-1, self.nof_predicates))
            

                # set predicate loss
                self.predicate_ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=shaped_labels_predicate,
                                                                             logits=shaped_belief_predicate,
                                                                             name="predicate_ce_loss")
                #self.predicate_original_ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=shaped_labels_predicate,
                #                                                                      logits=shaped_in_belief_predicate,
                #                                                                      name="predicate_original_ce_loss")

                #self.predicate_delta_loss = tf.maximum(self.predicate_ce_loss - self.predicate_original_ce_loss, 0.0)
                #self.predicate_delta_sum = tf.reduce_sum(tf.square(self.predicate_delta), axis=1)

                # set loss per requested loss_func
                if self.loss_func == "all":
                    self.loss_predicate = self.predicate_delta_loss + self.predicate_ce_loss + 0.01 * self.predicate_delta_sum
                elif self.loss_func == "exclude_sum":
                    self.loss_predicate = self.predicate_delta_loss + self.predicate_ce_loss
                elif self.loss_func == "ce":
                    self.loss_predicate = self.predicate_ce_loss
            
                self.loss_predicate_weighted = tf.multiply(self.loss_predicate, self.labels_coeff_loss_ph)

                img_weight = tf.reduce_sum(self.labels_coeff_loss_ph, name="img_weight")
                loss += tf.reduce_sum(self.loss_predicate_weighted) / ((self.rnn_steps - rnn_step) * img_weight)  

                # set object loss
                if self.including_object:
                    self.object_ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_object_ph,
                                                                                 logits=self.out_belief_object_lst[rnn_step],
                                                                                 name="object_ce_loss")

                    #self.object_original_ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_object_ph,
                    #                                                                      logits=self.belief_object_ph,
                    #                                                                      name="in_loss_predicate")

                    #self.object_delta_loss = tf.maximum(self.object_ce_loss - self.object_original_ce_loss, 0.0)
                    #self.object_delta_sum = tf.reduce_sum(tf.square(self.object_delta), axis=1)

                    # set loss per requested loss_func
                    if self.loss_func == "all":
                        self.loss_object = self.object_delta_loss + self.object_ce_loss + 0.01 * self.object_delta_sum
                    elif self.loss_func == "exclude_sum":
                        self.loss_object = self.object_delta_loss + self.object_ce_loss
                    elif self.loss_func == "ce":
                        self.loss_object = self.object_ce_loss

                    loss += self.lr_object_coeff * tf.reduce_mean(self.loss_object) / (self.rnn_steps - rnn_step)

            
            # reg
            #trainable_vars   = tf.trainable_variables() 
            #lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars ]) * 0.0001
            #loss += lossL2

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
        return self.out_predicate_probes, self.out_object_probes

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
