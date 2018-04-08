import sys

from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_50, bottleneck, resnet_v2

sys.path.append("..")

from keras.layers import GlobalAveragePooling2D, Dense

from FeaturesExtraction.Lib.Zoo import ModelZoo
from FilesManager.FilesManager import FilesManager
from Utils.Logger import Logger
import tensorflow as tf
import numpy as np
from vgg16 import vgg16


class End2EndModel(object):
    """
    End to End Module consists of Detector phase and Deep Belief Module which gets as an input the image and propogates
    the belief of predicates and objects. The E2E Module outputs an improved belief for predicates and objects
    """

    def __init__(self, gpi_type="FeatureAttention", nof_predicates=51, nof_objects=150, rnn_steps=1, is_train=True,
                 learning_rate=0.0001,
                 learning_rate_steps=1000, learning_rate_decay=0.5,
                 including_object=False, layers=[500, 500, 500], reg_factor=0.0, lr_object_coeff=4, config=None):
        """
        Construct module:
        - create input placeholders
        - apply SGP rnn_steps times
        - create labels placeholders
        - create module loss and train_step

        :type gpi_type: "Linguistic", "FeatureAttention", "NeighbourAttention"
        :param nof_predicates: nof predicate labels
        :param nof_objects: nof object labels
        :param rnn_steps: number of time to apply SGP
        :param is_train: whether the module will be used to train or eval
        """
        # save input
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_steps = learning_rate_steps
        self.learning_rate = learning_rate
        self.nof_predicates = nof_predicates
        self.nof_objects = nof_objects
        self.is_train = is_train
        self.rnn_steps = rnn_steps
        self.embed_size = 300
        self.gpi_type = gpi_type

        self.including_object = including_object
        self.lr_object_coeff = lr_object_coeff
        self.layers = layers
        self.reg_factor = reg_factor
        self.activation_fn = tf.nn.relu
        self.reuse = None
        self.config = config
        # logging module
        logger = Logger()

        # Create tf Graph Inputs
        self.create_placeholders()

        # Feature Extractor
        self.feature_extractor()
        # SGP
        self.sgp()

    def create_placeholders(self, scope_name="placeholders"):
        """
        This function creates the place holders for input and labels
        """
        with tf.variable_scope(scope_name):
            # todo: clean
            self.img_inputs_ph = tf.placeholder(shape=[None, self.config.crop_height, self.config.crop_width, 5],
                                                dtype=tf.float32, name="img_inputs")
            # size of slices of image relations (to avoid from OOM error)
            self.slices_size_ph = tf.placeholder(dtype=tf.int32, shape=[3])
            self.slices = tf.split(self.img_inputs_ph, self.slices_size_ph)

            # shape to be used by feature collector
            self.num_objects_ph = tf.placeholder(dtype=tf.int32, shape=[], name="num_of_objects_ph")

            ##
            # SGP input

            self.phase_ph = tf.placeholder(tf.bool, name='phase')

            # confidence
            # self.confidence_relation_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, self.nof_predicates),
            #                                             name="confidence_relation")
            # self.confidence_relation_ph = tf.contrib.layers.dropout(self.confidence_relation_ph, keep_prob=0.9, is_training=self.phase_ph)
            self.confidence_entity_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.nof_objects),
                                                       name="confidence_entity")
            # self.confidence_entity_ph = tf.contrib.layers.dropout(self.confidence_entity_ph, keep_prob=0.9, is_training=self.phase_ph)
            # spatial features
            self.entity_bb_ph = tf.placeholder(dtype=tf.float32, shape=(None, 4), name="obj_bb")

            # word embeddings
            # self.word_embed_entities_ph = tf.placeholder(dtype=tf.float32, shape=(self.nof_objects, self.embed_size),
            #                                             name="word_embed_objects")
            # self.word_embed_relations_ph = tf.placeholder(dtype=tf.float32,
            #                                              shape=(self.nof_predicates, self.embed_size),
            #                                              name="word_embed_predicates")

            # labels
            if self.is_train:
                self.labels_relation_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, self.nof_predicates),
                                                         name="labels_predicate")
                self.labels_entity_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.nof_objects),
                                                       name="labels_object")
                self.labels_coeff_loss_ph = tf.placeholder(dtype=tf.float32, shape=(None), name="labels_coeff_loss")

    def feature_extractor(self):
        """
        This function creates weights and biases
        """
        # Create First part
        # self.create_detection_net()
        # self.create_vgg_detection_net()
        self.create_resnet_detection_net()

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
            # todo: clean didn't succeed
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

    def create_vgg_detection_net(self):
        output_slices = []
        reuse = None
        # for slice in self.slices:
        #    relation_net = vgg16(slice, reuse=reuse)
        #    reuse = True
        #    output_slices.append(relation_net.features)

        # self.output_resnet50 = tf.concat(output_slices, 0)

        relation_net = vgg16(self.img_inputs_ph)
        self.output_resnet50 = relation_net.features

        N = tf.slice(tf.shape(self.confidence_entity_ph), [0], [1], name="N")
        M = tf.constant([50], dtype=tf.int32)
        relations_shape = tf.concat((N, N, M), 0)
        self.output_resnet50_reshaped = tf.reshape(self.output_resnet50, relations_shape)

    def create_resnet_detection_net(self):
        #reuse = None
        #output_slices = []
        #for slice in self.slices:
        #    slice_output, end_points = alexnet_v2(slice, num_classes=50, reuse=reuse)
        #    reuse = True
        #    output_slices.append(slice_output)

        #self.output_resnet50 = tf.concat(output_slices, 0)

        self.output_resnet50, end_points = self.resnet_v2_18(self.img_inputs_ph, num_classes=50)
        N = tf.slice(tf.shape(self.confidence_entity_ph), [0], [1], name="N")
        M = tf.constant([50], dtype=tf.int32)
        relations_shape = tf.concat((N, N, M), 0)
        self.output_resnet50_reshaped = tf.reshape(self.output_resnet50, relations_shape)

    def sgp(self):
        # store all the outputs of of rnn steps
        self.out_confidence_entity_lst = []
        self.out_confidence_relation_lst = []
        # rnn stage module
        confidence_relation = self.output_resnet50_reshaped
        confidence_entity = self.confidence_entity_ph

        # features msg
        for step in range(self.rnn_steps):
            confidence_relation, confidence_entity_temp = \
                self.sgp_cell(relation_features=confidence_relation,
                              entity_features=confidence_entity,
                              scope_name="deep_graph")
            # store the confidence
            self.out_confidence_relation_lst.append(confidence_relation)
            if self.including_object:
                confidence_entity = confidence_entity_temp
                # store the confidence
                self.out_confidence_entity_lst.append(confidence_entity_temp)
            self.reuse = True

        # confidence_entity = confidence_entity_temp
        self.out_confidence_relation = confidence_relation
        self.out_confidence_entity = confidence_entity
        reshaped_relation_confidence = tf.reshape(confidence_relation, (-1, self.nof_predicates))
        self.reshaped_relation_probes = tf.nn.softmax(reshaped_relation_confidence)
        self.out_relation_probes = tf.reshape(self.reshaped_relation_probes, tf.shape(confidence_relation),
                                              name="out_relation_probes")
        self.out_entity_probes = tf.nn.softmax(confidence_entity, name="out_entity_probes")

        # loss
        if self.is_train:
            # Learning rate
            self.lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name="lr_ph")

            self.loss, self.gradients, self.grad_placeholder, self.train_step = self.module_loss()

    def nn(self, features, layers, out, scope_name, seperated_layer=False, last_activation=None):
        """
        simple nn to convert features to confidence
        :param features: list of features tensor
        :param layers: hidden layers
        :param separated_layer: First run FC one each feature tensor separately
        :param out: output shape (used to reshape to required output shape)
        :param scope_name: tensorflow scope name
        :param last_activation: activation function for the last layer (None means no activation)
        :return: confidence
        """
        with tf.variable_scope(scope_name) as scope:

            # first layer each feature separately
            features_h_lst = []
            index = 0
            for feature in features:
                if seperated_layer:
                    in_size = feature.shape[-1]._value
                    scope = str(index)
                    h = tf.contrib.layers.fully_connected(feature, in_size, reuse=self.reuse, scope=scope,
                                                          activation_fn=self.activation_fn)
                    index += 1
                    features_h_lst.append(h)
                else:
                    features_h_lst.append(feature)

            h = tf.concat(features_h_lst, axis=-1)
            h = tf.contrib.layers.dropout(h, keep_prob=0.9, is_training=self.phase_ph)
            for layer in layers:
                scope = str(index)
                h = tf.contrib.layers.fully_connected(h, layer, reuse=self.reuse, scope=scope,
                                                      activation_fn=self.activation_fn)
                h = tf.contrib.layers.dropout(h, keep_prob=0.9, is_training=self.phase_ph)
                index += 1

            scope = str(index)
            y = tf.contrib.layers.fully_connected(h, out, reuse=self.reuse, scope=scope, activation_fn=last_activation)
        return y

    def sgp_cell(self, relation_features, entity_features, scope_name="rnn_cell"):
        """
        SGP step - which get as an input a confidence of the predicates and objects and return an improved confidence of the predicates and the objects
        :return:
        :param relation_features: in relation confidence
        :param entity_features: in entity confidence
        :param scope_name: sgp step scope
        :return: improved relation probabilities, improved relation confidence,  improved entity probabilities and improved entity confidence
        """
        with tf.variable_scope(scope_name):

            # add the spatial features to entity features
            orig_entity_features = entity_features
            entity_features = tf.concat((entity_features, self.entity_bb_ph), axis=1)

            # word embeddings
            # expand object word embed
            N = tf.slice(tf.shape(entity_features), [0], [1], name="N")

            # append relations in both directions
            self.relation_features = tf.concat((relation_features, tf.transpose(relation_features, perm=[1, 0, 2])),
                                               axis=2)

            # expand object confidence
            self.extended_entity_features_shape = tf.concat((N, tf.shape(entity_features)), 0)
            self.expand_object_features = tf.add(tf.zeros(self.extended_entity_features_shape),
                                                 entity_features,
                                                 name="expand_object_features")
            # expand subject confidence
            self.expand_subject_features = tf.transpose(self.expand_object_features, perm=[1, 0, 2],
                                                        name="expand_subject_features")

            ##
            # Node Neighbours
            self.object_ngbrs = [self.expand_object_features, self.expand_subject_features, relation_features]
            # apply phi
            self.object_ngbrs_phi = self.nn(features=self.object_ngbrs, layers=[], out=500, scope_name="nn_phi")
            # Attention mechanism
            if self.gpi_type == "FeatureAttention":
                self.object_ngbrs_scores = self.nn(features=self.object_ngbrs, layers=[], out=500,
                                                   scope_name="nn_phi_atten")
                self.object_ngbrs_weights = tf.nn.softmax(self.object_ngbrs_scores, dim=1)
                self.object_ngbrs_phi_all = tf.reduce_sum(tf.multiply(self.object_ngbrs_phi, self.object_ngbrs_weights),
                                                          axis=1)

            elif self.gpi_type == "NeighbourAttention":
                self.object_ngbrs_scores = self.nn(features=self.object_ngbrs, layers=[], out=1,
                                                   scope_name="nn_phi_atten")
                self.object_ngbrs_weights = tf.nn.softmax(self.object_ngbrs_scores, dim=1)
                self.object_ngbrs_phi_all = tf.reduce_sum(tf.multiply(self.object_ngbrs_phi, self.object_ngbrs_weights),
                                                          axis=1)
            else:
                self.object_ngbrs_phi_all = tf.reduce_mean(self.object_ngbrs_phi, axis=1)

            ##
            # Nodes
            self.object_ngbrs2 = [entity_features, self.object_ngbrs_phi_all]
            # apply alpha
            self.object_ngbrs2_alpha = self.nn(features=self.object_ngbrs2, layers=[], out=500, scope_name="nn_phi2")
            # Attention mechanism
            if self.gpi_type == "FeatureAttention":
                self.object_ngbrs2_scores = self.nn(features=self.object_ngbrs2, layers=[], out=500,
                                                    scope_name="nn_phi2_atten")
                self.object_ngbrs2_weights = tf.nn.softmax(self.object_ngbrs2_scores, dim=0)
                self.object_ngbrs2_alpha_all = tf.reduce_sum(
                    tf.multiply(self.object_ngbrs2_alpha, self.object_ngbrs2_weights), axis=0)
            elif self.gpi_type == "NeighbourAttention":
                self.object_ngbrs2_scores = self.nn(features=self.object_ngbrs2, layers=[], out=1,
                                                    scope_name="nn_phi2_atten")
                self.object_ngbrs2_weights = tf.nn.softmax(self.object_ngbrs2_scores, dim=0)
                self.object_ngbrs2_alpha_all = tf.reduce_sum(
                    tf.multiply(self.object_ngbrs2_alpha, self.object_ngbrs2_weights), axis=0)
            else:
                self.object_ngbrs2_alpha_all = tf.reduce_mean(self.object_ngbrs2_alpha, axis=0)

            expand_graph_shape = tf.concat((N, N, tf.shape(self.object_ngbrs2_alpha_all)), 0)
            expand_graph = tf.add(tf.zeros(expand_graph_shape), self.object_ngbrs2_alpha_all)

            ##
            # rho relation (relation prediction)
            # The input is object features, subject features, relation features and the representation of the graph
            self.expand_obj_ngbrs_phi_all = tf.add(tf.zeros_like(self.object_ngbrs_phi), self.object_ngbrs_phi_all)
            self.expand_sub_ngbrs_phi_all = tf.transpose(self.expand_obj_ngbrs_phi_all, perm=[1, 0, 2])
            self.relation_all_features = [relation_features, self.expand_object_features, self.expand_subject_features,
                                          expand_graph]

            pred_delta = self.nn(features=self.relation_all_features, layers=self.layers, out=self.nof_predicates,
                                 scope_name="nn_pred")
            pred_forget_gate = self.nn(features=self.relation_all_features, layers=[], out=1,
                                       scope_name="nn_pred_forgate", last_activation=tf.nn.sigmoid)
            out_confidence_relation = pred_delta  # + pred_forget_gate * relation_features

            ##
            # rho entity (entity prediction)
            # The input is entity features, entity neighbour features and the representation of the graph
            if self.including_object:
                self.object_all_features = [entity_features, expand_graph[0], self.object_ngbrs_phi_all]
                obj_delta = self.nn(features=self.object_all_features, layers=self.layers, out=self.nof_objects,
                                    scope_name="nn_obj")
                obj_forget_gate = self.nn(features=self.object_all_features, layers=[], out=self.nof_objects,
                                          scope_name="nn_obj_forgate", last_activation=tf.nn.sigmoid)
                out_confidence_object = obj_delta + obj_forget_gate * orig_entity_features
            else:
                out_confidence_object = orig_entity_features

            return out_confidence_relation, out_confidence_object

    def module_loss(self, scope_name="loss"):
        """
        SGP loss
        :param scope_name: tensor flow scope name
        :return: loss and train step
        """
        with tf.variable_scope(scope_name):
            # reshape to batch like shape
            shaped_labels_predicate = tf.reshape(self.labels_relation_ph, (-1, self.nof_predicates))

            # relation gt
            self.gt = tf.argmax(shaped_labels_predicate, axis=1)

            loss = 0

            for rnn_step in range(self.rnn_steps):

                shaped_confidence_predicate = tf.reshape(self.out_confidence_relation_lst[rnn_step],
                                                         (-1, self.nof_predicates))

                # set predicate loss
                self.relation_ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=shaped_labels_predicate,
                                                                                logits=shaped_confidence_predicate,
                                                                                name="relation_ce_loss")

                self.loss_relation = self.relation_ce_loss
                self.loss_relation_weighted = tf.multiply(self.loss_relation, self.labels_coeff_loss_ph)

                loss += tf.reduce_sum(self.loss_relation_weighted)

                # set object loss
                if self.including_object:
                    self.object_ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_entity_ph,
                                                                                  logits=self.out_confidence_entity_lst[
                                                                                      rnn_step],
                                                                                  name="object_ce_loss")

                    loss += self.lr_object_coeff * tf.reduce_sum(self.object_ce_loss)

            # reg
            trainable_vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars]) * self.reg_factor
            loss += lossL2

            # minimize
            # opt = tf.train.GradientDescentOptimizer(self.lr_ph)
            opt = tf.train.AdamOptimizer(self.lr_ph)
            # opt = tf.train.MomentumOptimizer(self.lr_ph, 0.9, use_nesterov=True)
            gradients = opt.compute_gradients(loss)
            # create placeholder to minimize in a batch
            grad_placeholder = [(tf.placeholder("float", shape=grad[0].get_shape()), grad[1]) for grad in gradients]

            train_step = opt.apply_gradients(grad_placeholder)
        return loss, gradients, grad_placeholder, train_step

    def get_in_ph(self):
        """
        get input place holders
        """
        return self.confidence_entity_ph, self.entity_bb_ph

    def get_output(self):
        """
        get module output
        """
        return self.out_relation_probes, self.out_entity_probes

    def get_labels_ph(self):
        """
        get module labels ph (used for train)
        """
        return self.labels_relation_ph, self.labels_entity_ph, self.labels_coeff_loss_ph

    def get_module_loss(self):
        """
        get module loss and train step
        """
        return self.loss, self.gradients, self.grad_placeholder, self.train_step

    def resnet_v2_18(self,
                     inputs,
                     num_classes=None,
                     global_pool=True,
                     output_stride=None,
                     reuse=None,
                     scope='resnet_v2_50'):
        """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
        blocks = [
            resnet_utils.Block('block1', bottleneck,
                               [(64, 64, 3)] * 1 + [(64, 64, 3)]),
            resnet_utils.Block('block2', bottleneck,
                               [(128, 128, 3)] * 1 + [(128, 128, 3)]),
            resnet_utils.Block('block3', bottleneck,
                               [(256, 256, 3)] * 1 + [(256, 256, 3)]),
            resnet_utils.Block('block4', bottleneck, [(512, 512, 3)] * 1 + [(512, 512, 3)])
        ]
        return resnet_v2(
            inputs,
            blocks,
            num_classes,
            global_pool,
            output_stride,
            include_root_block=True,
            reuse=reuse,
            scope=scope)

