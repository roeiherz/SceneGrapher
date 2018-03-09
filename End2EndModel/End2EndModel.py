from keras.layers import GlobalAveragePooling2D, Dense

from FeaturesExtraction.Lib.Zoo import ModelZoo
from FilesManager.FilesManager import FilesManager
from Utils.Logger import Logger
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

__author__ = 'roeih'


class End2EndModel(object):
    """
    End to End Module consists of Detector phase and Deep Belief Module which gets as an input the image and propogates
    the belief of predicates and objects. The E2E Module outputs an improved belief for predicates and objects
    """

    def __init__(self, nof_predicates, nof_objects, visual_features_predicate_size, visual_features_object_size,
                 rnn_steps=2, is_train=True, loss_func="all", learning_rate=0.1, learning_rate_steps=1000,
                 learning_rate_decay=0.5, including_object=False, include_bb=False, layers=[1000, 200],
                 reg_factor=0.03, lr_object_coeff=1, config=None):
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
        self.embed_size = 300
        self.loss_func = loss_func
        self.including_object = including_object
        self.lr_object_coeff = lr_object_coeff
        self.layers = layers
        self.include_bb = include_bb
        self.reg_factor = reg_factor
        self.activation_fn = tf.nn.relu
        self.config = config

        # logging module
        logger = Logger()

        # Create tf Graph Inputs
        self.create_placeholders()

        # Create Weights
        self.create_weights()

        # @todo: clean
        # # Linear activation, using rnn inner loop last output
        # h1 = tf.matmul(self.output_resnet50, self.nn_predicate_w_1) + self.nn_predicate_b_1
        # self.logits = tf.matmul(h1, self.nn_predicate_w_5) + self.nn_predicate_b_5
        # # self.logits = self.output_resnet50

        # store all the outputs of of rnn steps
        self.out_belief_object_lst = []
        self.out_belief_predicate_lst = []
        # rnn stage module
        belief_predicate = self.output_resnet50_reshaped
        belief_object = self.belief_object_ph

        # features msg
        for step in range(self.rnn_steps):
            belief_predicate, belief_object_temp = \
                self.deep_graph(in_belief_predicate=belief_predicate,
                                in_belief_object=belief_object,
                                in_extended_belief_object_shape=self.extended_belief_object_shape_ph,
                                scope_name="deep_graph")
            # store the belief
            self.out_belief_predicate_lst.append(belief_predicate)
            if self.including_object:
                belief_object = belief_object_temp
                # store the belief
                self.out_belief_object_lst.append(belief_object_temp)

        self.out_belief_predicate = belief_predicate
        self.out_belief_object = belief_object
        reshaped_predicate_belief = tf.reshape(belief_predicate, (-1, self.nof_predicates))
        self.reshaped_predicete_probes = tf.nn.softmax(reshaped_predicate_belief)
        self.out_predicate_probes = tf.reshape(self.reshaped_predicete_probes, tf.shape(belief_predicate),
                                               name="out_predicate_probes")
        self.out_object_probes = tf.nn.softmax(belief_object, name="out_object_probes")

        # loss
        if self.is_train:
            # Learning rate
            self.lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name="lr_ph")

            self.loss, self.gradients, self.grad_placeholder, self.train_step = self.module_loss()

    def create_placeholders(self, scope_name="placeholders"):
        """
        This function creates the place holders for input and labels
        """
        with tf.variable_scope(scope_name):
            # todo: clean
            # self.img_inputs_ph = tf.placeholder(shape=[None, self.config.crop_height, self.config.crop_width, 5],
            #                                 dtype=tf.float32, name="img_inputs")
            # self.img_labels_ph = tf.placeholder(shape=[None, self.num_classes], dtype=tf.float32,
            #                                 name="image_output")
            self.img_inputs_ph = tf.contrib.keras.layers.Input(
                shape=(self.config.crop_height, self.config.crop_width, 5), name="image_input_ph")

            # shape to be used by feature collector
            self.num_objects_ph = tf.placeholder(dtype=tf.int32, shape=[], name="num_of_objects_ph")

            ## module input
            # batch normalization
            self.phase_ph = tf.placeholder(tf.bool, name='phase')

            # Visual features
            self.visual_features_predicate_ph = tf.placeholder(dtype=tf.float32,
                                                               shape=(None, None, self.visual_features_predicate_size),
                                                               name="visual_feautres_predicate")
            self.visual_features_object_ph = tf.placeholder(dtype=tf.float32,
                                                            shape=(None, self.visual_features_object_size),
                                                            name="visual_features_object")
            # belief
            # self.belief_predicate_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, self.nof_predicates),
            #                                           name="belief_predicate")
            self.belief_object_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.nof_objects),
                                                   name="belief_object")

            # shape to be used by feature collector
            self.extended_belief_object_shape_ph = tf.placeholder(dtype=tf.int32, shape=(3),
                                                                  name="extended_belief_object_shape")

            # shape to be used by feature collector
            self.extended_obj_bb_shape_ph = tf.placeholder(dtype=tf.int32, shape=(3), name="extended_obj_bb_shape")
            self.obj_bb_ph = tf.placeholder(dtype=tf.float32, shape=(None, 4), name="obj_bb")
            self.expand_obj_bb = tf.add(tf.zeros(self.extended_obj_bb_shape_ph), self.obj_bb_ph, name="expand_obj_bb")
            # expand subject bb
            expand_sub_bb = tf.transpose(self.expand_obj_bb, perm=[1, 0, 2], name="expand_sub_bb")
            self.expand_sub_bb = expand_sub_bb
            bb_features_0 = tf.concat((expand_sub_bb, self.expand_obj_bb), axis=2, name="bb_features")
            self.bb_features = bb_features_0
            # word embeddings
            self.word_embed_objects = tf.placeholder(dtype=tf.float32, shape=(self.nof_objects, self.embed_size),
                                                     name="word_embed_objects")
            self.word_embed_predicates = tf.placeholder(dtype=tf.float32, shape=(self.nof_predicates, self.embed_size),
                                                        name="word_embed_predicates")

            # labels
            if self.is_train:
                self.labels_predicate_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, self.nof_predicates),
                                                          name="labels_predicate")
                self.labels_object_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.nof_objects),
                                                       name="labels_object")
                self.labels_coeff_loss_ph = tf.placeholder(dtype=tf.float32, shape=(None), name="labels_coeff_loss")

    def create_weights(self):
        """
        This function creates weights and biases
        """
        # Create First part
        self.create_detection_net()

        #todo: clean
        # Create Second part
        # self.create_deep_belief_net()

    #todo: clean
    def create_deep_belief_net(self, scope_name="deep_belief"):
        """
        This function creates weights and biases
        """
        h1_size = 500
        h4_size = 500
        out_size = 51

        with tf.variable_scope(scope_name):
            # Define the initialization of the first layer
            self.nn_predicate_w_1 = tf.get_variable(name="w1", shape=(51, h1_size),
                                                    initializer=tf.truncated_normal_initializer(stddev=0.03))
            self.nn_predicate_b_1 = tf.get_variable(name="b1", shape=(h1_size),
                                                    initializer=tf.truncated_normal_initializer(stddev=0.03))
            # Define the initialization of the layer 5
            self.nn_predicate_w_5 = tf.get_variable(name="w5", shape=(h4_size, out_size),
                                                    initializer=tf.truncated_normal_initializer(stddev=0.03))
            self.nn_predicate_b_5 = tf.get_variable(name="b5", shape=(out_size),
                                                    initializer=tf.truncated_normal_initializer(stddev=0.03))

    def create_detection_net(self, scope_name="detector"):
        """
        This function creates weights and biases for the detection architecture unit
        """
        with tf.variable_scope(scope_name):
            # Define ResNet50 model With Top
            net = ModelZoo()
            model_resnet50 = net.resnet50_with_masking_dual(self.img_inputs_ph,
                                                            trainable=self.config.resnet_body_trainable)
            model_resnet50 = GlobalAveragePooling2D(name='global_avg_pool')(model_resnet50)
            self.output_resnet50 = Dense(self.nof_predicates, kernel_initializer="he_normal", activation=None,
                                         name='fc')(model_resnet50)
            self.output_resnet50_reshaped = tf.reshape(self.output_resnet50, [self.num_objects_ph, self.num_objects_ph,
                                                                              self.nof_predicates])
            # todo: clean didn't successeed
            # # Set negative predicate in diagonal
            # self.try1 = tf.constant([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            #                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            #                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            #                            0.0, 0.0, 1.0]]])
            # self.try4 = tf.constant([[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            #                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            #                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            #                            1.0, 1.0, 0.0]]])
            # self.try2 = tf.matrix_set_diag(self.belief_masked, tf.ones([6, 6]))
            # self.try3 = tf.tile(self.try1, [self.num_objects_ph, 1])
            # # mask = tf.transpose(tf.matrix_diag(tf.ones_like(tf.transpose(self.output_resnet50_reshaped)))) * -1 + 1
            # # mask = tf.less(self.output_resnet50, 0.0001 * tf.ones_like(original_tensor))
            mask = tf.transpose(tf.matrix_diag(tf.ones_like(tf.transpose(self.output_resnet50_reshaped[0])))) * -1 + 1
            self.output_resnet50_reshaped = tf.multiply(self.output_resnet50_reshaped, mask)

    def deep_graph(self, in_belief_predicate, in_belief_object,
                   in_extended_belief_object_shape, scope_name="rnn_cell"):
        """
        RNN stage - which get as an input a belief of the predicates and objects and return an improved belief of the predicates and the objects
        :return:
        :param in_belief_predicate: predicate belief of the last stage in the RNN
        :param in_belief_object: object belief of the last stage in the RNNS
        :param in_extended_belief_object_shape: the shape of the extended version of object belief (N, N, NOF_OBJECTS)
        :param scope_name: rnn stage scope
        :return: improved predicates probabilities, improved predicate belief,  improved object probabilities and improved object belief
        """
        with tf.variable_scope(scope_name):

            in_belief_predicate = in_belief_predicate - tf.reduce_mean(self.output_resnet50_reshaped, axis=2,
                                                                       keep_dims=True)
            self.in_belief_predicate = in_belief_predicate
            in_belief_object = in_belief_object - tf.reduce_mean(in_belief_object, axis=1, keep_dims=True)
            self.in_belief_object = in_belief_object

            # belief to probes
            self.in_belief_predicate_actual = in_belief_predicate
            predicate_probes = tf.nn.softmax(in_belief_predicate)
            self.predicate_probes = predicate_probes
            in_belief_predicate_norm = tf.log(predicate_probes + tf.constant(1e-10))

            in_belief_object = self.belief_object_ph
            self.in_belief_object_actual = in_belief_object
            object_probes = tf.nn.softmax(in_belief_object)
            self.object_probes = object_probes
            in_belief_object_norm = tf.log(object_probes + tf.constant(1e-10))

            # word embeddings
            # expand object embed
            self.obj_prediction = tf.argmax(self.object_probes, axis=1)
            self.obj_prediction_val = tf.reduce_max(self.object_probes, axis=1)
            self.embed_objects = tf.gather(self.word_embed_objects, self.obj_prediction)
            self.embed_objects = tf.transpose(tf.multiply(tf.transpose(self.embed_objects), self.obj_prediction_val))
            in_extended_belief_embed_shape = in_extended_belief_object_shape - tf.constant(
                [0, 0, 150 - self.embed_size])
            expand_embed_object = tf.add(tf.zeros(in_extended_belief_embed_shape), self.embed_objects)
            in_belief_object_norm = self.embed_objects

            # expand subject belief
            expand_embed_subject = tf.transpose(expand_embed_object, perm=[1, 0, 2])
            self.expand_embed_subject = expand_embed_subject
            self.pred_prediction = tf.argmax(self.predicate_probes, axis=2)
            self.pred_prediction_val = tf.reduce_max(self.predicate_probes, axis=2)
            self.embed_predicates = tf.gather(self.word_embed_predicates, tf.reshape(self.pred_prediction, [-1]))
            self.embed_predicates = tf.transpose(
                tf.multiply(tf.transpose(self.embed_predicates), tf.reshape(self.pred_prediction_val, [-1])))
            self.embed_predicates = tf.reshape(self.embed_predicates, tf.shape(expand_embed_object))
            in_belief_predicate_norm = self.embed_predicates
            # expand to NxN
            self.predicate_opposite = tf.transpose(in_belief_predicate_norm, perm=[1, 0, 2])
            # expand object belief
            self.expand_object_belief = tf.add(tf.zeros(in_extended_belief_embed_shape), in_belief_object_norm,
                                               name="expand_object_belief")
            # expand subject belief
            self.expand_subject_belief = tf.transpose(self.expand_object_belief, perm=[1, 0, 2],
                                                      name="expand_subject_belief")

            N = tf.slice(tf.shape(in_belief_predicate_norm), [0], [1], name="N")
            expand_predicate_3d_shape = tf.concat((N, tf.shape(in_belief_predicate_norm)), 0)
            expand_object_3d_shape = tf.concat((N, N, tf.shape(in_belief_object_norm)), 0)
            in_belief_predicate_expand_3d = tf.add(tf.zeros(expand_predicate_3d_shape), in_belief_predicate_norm)
            in_belief_predicate_opp_expand_3d = tf.transpose(in_belief_predicate_expand_3d, perm=[0, 2, 1, 3])
            self.predicate_expand_3d = tf.transpose(in_belief_predicate_expand_3d, perm=[1, 2, 0, 3])
            self.object_expand_3d = tf.add(tf.zeros(expand_object_3d_shape), self.expand_object_belief)
            self.subject_expand_3d = tf.transpose(self.object_expand_3d, perm=[0, 2, 1, 3])
            self.object_atten_expand_3d = tf.transpose(self.object_expand_3d, perm=[2, 0, 1, 3])
            # in_belief_object_expand_3d = tf.transpose(in_belief_object_expand_3d, perm=[2, 0, 1, 3])
            # self.expand_object_ngbrs_phi_all_3d = tf.add(tf.zeros(expand_object_3d_shape), self.expand_object_ngbrs_phi_all)
            # self.expand_object_ngbrs_phi_all_3d = tf.transpose(self.expand_object_ngbrs_phi_all_3d, perm=[2, 0, 1, 3])
            expand_bb_3d_shape = tf.concat((N, tf.shape(self.bb_features)), 0)
            self.pred_bb_expand_3d = tf.add(tf.zeros(expand_bb_3d_shape), self.bb_features)
            self.pred2_bb_expand_3d = tf.transpose(self.pred_bb_expand_3d, perm=[1, 2, 0, 3])

            # stage 1 graph embedding
            self.object_ngbrs = [self.expand_object_belief, self.expand_subject_belief, in_belief_predicate_norm,
                                 self.predicate_opposite, self.bb_features]

            self.object_ngbrs_phi = self.nn(features=self.object_ngbrs, layers=[], out=500, scope_name="nn_phi")

            self.object_ngbrs_scores = self.nn(features=self.object_ngbrs, layers=[], out=500,
                                               scope_name="nn_phi_atten")
            self.object_ngbrs_weights = tf.nn.softmax(self.object_ngbrs_scores, dim=1)
            self.object_ngbrs_phi_all = tf.reduce_sum(tf.multiply(self.object_ngbrs_phi, self.object_ngbrs_weights),
                                                      axis=1)
            self.expand_obj_ngbrs_phi_all = tf.add(tf.zeros_like(self.object_ngbrs_phi), self.object_ngbrs_phi_all)
            self.expand_sub_ngbrs_phi_all = tf.transpose(self.expand_obj_ngbrs_phi_all, perm=[1, 0, 2])
            # stage 2 graph embedding

            self.object_ngbrs2 = [in_belief_object_norm, self.object_ngbrs_phi_all]
            self.object_ngbrs2_phi = self.nn(features=self.object_ngbrs2, layers=[], out=500, scope_name="nn_phi2")
            self.object_ngbrs2_scores = self.nn(features=self.object_ngbrs2, layers=[], out=500,
                                                scope_name="nn_phi2_atten")
            self.object_ngbrs2_weights = tf.nn.softmax(self.object_ngbrs2_scores, dim=0)
            self.object_ngbrs2_phi_all = tf.reduce_sum(tf.multiply(self.object_ngbrs2_phi, self.object_ngbrs2_weights),
                                                       axis=0)
            expand_graph_shape = tf.concat((N, N, tf.shape(self.object_ngbrs2_phi_all)), 0)
            expand_graph = tf.add(tf.zeros(expand_graph_shape), self.object_ngbrs2_phi_all)
            # predicate refine
            self.predicate_all_features = [self.predicate_probes, self.expand_object_belief, self.expand_subject_belief,
                                           self.expand_sub_ngbrs_phi_all, self.expand_obj_ngbrs_phi_all, expand_graph,
                                           self.bb_features]

            pred_delta = self.nn(features=self.predicate_all_features, layers=self.layers, out=self.nof_predicates,
                                 scope_name="nn_pred")
            pred_forget_gate = self.nn(features=self.predicate_all_features, layers=[], out=1,
                                       scope_name="nn_pred_forgate", last_activation=tf.nn.sigmoid)
            pred_input_gate = self.nn(features=self.predicate_all_features, layers=[], out=1,
                                      scope_name="nn_pred_ingate", last_activation=tf.nn.sigmoid)
            out_belief_predicate = pred_delta + pred_forget_gate * in_belief_predicate
            # object refine
            self.object_all_features = [self.object_probes, expand_graph[0], self.object_ngbrs_phi_all]
            if self.including_object:
                obj_delta = self.nn(features=self.object_all_features, layers=self.layers, out=self.nof_objects,
                                    scope_name="nn_obj")
                obj_forget_gate = self.nn(features=self.object_all_features, layers=[], out=self.nof_objects,
                                          scope_name="nn_obj_forgate", last_activation=tf.nn.sigmoid)
                out_belief_object = obj_delta + obj_forget_gate * in_belief_object
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

            # predicate gt
            self.gt = tf.argmax(shaped_labels_predicate, axis=1)

            # get negative predicate indices
            ones = tf.ones_like(shaped_labels_predicate)
            zeros = tf.zeros_like(shaped_labels_predicate)
            neg_const = tf.constant(self.nof_predicates - 1, dtype=tf.int64)
            self.neg_indices = tf.equal(self.gt, neg_const)

            # get postive predicate indices
            self.pos_indices = tf.not_equal(self.gt, neg_const)

            loss = 0

            for rnn_step in range(self.rnn_steps):

                shaped_belief_predicate = tf.reshape(self.out_belief_predicate_lst[rnn_step], (-1, self.nof_predicates))

                # set predicate loss
                self.predicate_ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=shaped_labels_predicate,
                                                                                 logits=shaped_belief_predicate,
                                                                                 name="predicate_ce_loss")

                self.shaped_scores = -tf.log(self.reshaped_predicete_probes)
                self.neg_scores = tf.where(self.neg_indices, self.shaped_scores,
                                           ones * tf.reduce_max(self.shaped_scores))
                self.pos_scores = tf.where(self.pos_indices, self.shaped_scores,
                                           ones * tf.reduce_max(self.shaped_scores))
                self.pos_score = tf.reduce_max(
                    tf.where(self.pos_indices, tf.multiply(self.shaped_scores, shaped_labels_predicate), zeros), axis=1)
                self.max_neg_score = tf.reduce_min(self.neg_scores[:, :self.nof_predicates - 1])
                self.wrong_score = tf.reduce_min(tf.multiply(self.shaped_scores, ones - shaped_labels_predicate),
                                                 axis=1)
                self.error_score = tf.minimum(self.wrong_score, self.max_neg_score)
                self.loss_predicate = tf.reduce_sum(tf.maximum(0.0, self.pos_score - self.error_score + 1.0))

                # self.loss_predicate = tf.reduce_sum(self.labels_coeff_loss_ph) * tf.maximum(0.0, 1.0 + self.max_pos_log_probe - self.min_neg_log_probe)
                # self.loss_predicate = tf.reduce_sum(self.loss_pos) + tf.reduce_sum(self.weighted_neg_loss)

                # loss += self.loss_predicate

                self.loss_predicate = self.predicate_ce_loss
                self.loss_predicate_weighted = tf.multiply(self.loss_predicate, self.labels_coeff_loss_ph)

                loss += tf.reduce_sum(self.loss_predicate_weighted)

                # set object loss
                if self.including_object:
                    self.object_ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_object_ph,
                                                                                  logits=self.out_belief_object_lst[
                                                                                      rnn_step],
                                                                                  name="object_ce_loss")

                    # object weights according to number of positives per object

                    # self.reshaped_weights = tf.reshape(self.labels_coeff_loss_ph, tf.shape(self.out_predicate_probes[:,:,0]))
                    self.reshaped_weights = tf.reshape(tf.to_float(self.pos_indices),
                                                       tf.shape(self.out_predicate_probes[:, :, 0]))
                    self.sub_weights = tf.reduce_sum(self.reshaped_weights, axis=1)
                    self.obj_weights = tf.reduce_sum(self.reshaped_weights, axis=0)
                    self.obj_total_weights = tf.add(self.sub_weights, self.obj_weights)
                    # self.obj_total_weights /= tf.reduce_sum(self.obj_total_weights)

                    self.loss_object = tf.multiply(self.object_ce_loss, self.obj_total_weights)

                    loss += self.lr_object_coeff * tf.reduce_sum(self.object_ce_loss)  # (self.loss_object)

            # reg
            trainable_vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars]) * self.reg_factor
            loss += lossL2

            # minimize
            opt = tf.train.GradientDescentOptimizer(self.lr_ph)
            # opt = tf.train.AdamOptimizer(self.lr_ph)
            # train_step = opt.minimize(loss)
            # opt = tf.train.MomentumOptimizer(self.lr_ph, 0.9, use_nesterov=True)
            # gradients = []
            # grad_placeholder = []
            gradients = opt.compute_gradients(loss)
            # create placeholder to minimize in a batch
            grad_placeholder = [(tf.placeholder("float", shape=grad[0].get_shape()), grad[1]) for grad in gradients]

            train_step = opt.apply_gradients(grad_placeholder)
        return loss, gradients, grad_placeholder, train_step

    def nn(self, features, layers, out, scope_name, seperated_layer=False, last_activation=None):
        """
        simple nn to convert features to belief
        :param features: features tensor
        :param out: output shape (used to reshape to required output shape)
        :param scope_name: tensorflow scope name
        :return: belief
        """
        with tf.variable_scope(scope_name):

            # first layer each feature seperatly
            features_h_lst = []
            for feature in features:
                if seperated_layer:
                    in_size = feature.shape[-1]._value
                    h = tf.contrib.layers.fully_connected(feature, in_size, activation_fn=self.activation_fn)
                    features_h_lst.append(h)
                else:
                    features_h_lst.append(feature)

            h = tf.concat(features_h_lst, axis=-1)

            index = 0
            for layer in layers:
                h = tf.contrib.layers.fully_connected(h, layer, activation_fn=self.activation_fn)
                index += 1

            y = tf.contrib.layers.fully_connected(h, out, activation_fn=last_activation)

        return y

    def get_in_ph(self):
        """
        get input place holders
        """
        return self.img_inputs_ph, self.belief_object_ph, self.extended_belief_object_shape_ph, \
               self.visual_features_predicate_ph, self.visual_features_object_ph, self.num_objects_ph

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
