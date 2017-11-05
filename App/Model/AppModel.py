import traceback
import itertools
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.models import Model
import numpy as np
from FeaturesExtraction.Lib.Config import Config
from FeaturesExtraction.Lib.VisualGenomeDataGenerator import visual_genome_data_cnn_generator_with_batch, \
    visual_genome_data_predicate_mask_dual_pairs_generator_with_batch, \
    visual_genome_data_predicate_pairs_generator_with_batch
from FeaturesExtraction.Lib.Zoo import ModelZoo
import math
import os
from FeaturesExtraction.Utils.Boxes import BOX, find_union_box
from FeaturesExtraction.Utils.Utils import get_img, get_mask_from_object, get_img_resize
from FilesManager.FilesManager import FilesManager
from Module.Module import Module
import tensorflow as tf

__author__ = 'roeih'


class AppModel(object):
    """
    This model represents a full real time model: Feature Extractor model + Belief RNN model
    """
    NUM_EPOCHS = 1
    NUM_BATCHES = 128 * 3
    # feature sizes
    VISUAL_FEATURES_PREDICATE_SIZE = 2048
    VISUAL_FEATURES_OBJECT_SIZE = 2048
    RNN_STEPS = 2

    def __init__(self, objects_model_weight_path="", predicates_model_weight_path="", gpu_num="0"):
        self.objects_model_weight_path = objects_model_weight_path
        self.predicates_model_weight_path = predicates_model_weight_path
        # Store modules
        self.objects_model = None
        self.predicates_model = None
        self.objects_no_activation_model = None
        self.predicates_no_activation_model = None
        self.noactivation_outputs_predicate_func = None
        self.noactivation_outputs_object_func = None
        self.rnn_belief_model = None
        self.session = None
        self.config = Config(gpu_num)
        # Set Jitter to False no matters
        self.config.jitter = False

    def load_feature_extractions_model(self, number_of_classes_objects, number_of_classes_predicates):
        """
        This function load Feature Extraction model
        :param number_of_classes_objects: number of labels of objects
        :param number_of_classes_predicates: number of labels of predicates
        :return:
        """

        # Load Models
        if self.objects_model is None:
            self.objects_model = self.get_model(number_of_classes_objects, weight_path=self.objects_model_weight_path)

        if self.predicates_model is None:
            self.predicates_model = self.get_model(number_of_classes_predicates,
                                                   weight_path=self.predicates_model_weight_path, use_mask=True)

        # Load Models without any activations
        if self.objects_no_activation_model is None:
            self.objects_no_activation_model = self.get_model(number_of_classes_objects,
                                                              weight_path=self.objects_model_weight_path,
                                                              activation=None)

        if self.predicates_no_activation_model is None:
            self.predicates_no_activation_model = self.get_model(number_of_classes_predicates,
                                                                 weight_path=self.predicates_model_weight_path,
                                                                 activation=None, use_mask=True)

        # Load Functions
        if self.noactivation_outputs_predicate_func is None:
            self.noactivation_outputs_predicate_func = K.function([self.predicates_no_activation_model.layers[0].input],
                                                                  [self.predicates_no_activation_model.layers[-1].output])

        if self.noactivation_outputs_object_func is None:
            self.noactivation_outputs_object_func = K.function([self.objects_no_activation_model.layers[0].input],
                                                               [self.objects_no_activation_model.layers[-1].output])

    def load_belief_rnn_model(self, number_of_classes_objects, number_of_classes_predicates,
                              module_name="dual_final3_best", module_name_predcl="dual_predcl_sum2_best"):
        """
        This function load Feature Extraction model
        :param number_of_classes_objects: number of labels of objects
        :param number_of_classes_predicates: number of labels of predicates
        :param module_name: the name of the module to load
        :return:
        """

        if self.rnn_belief_model is None:
            self.rnn_belief_model = Module(nof_predicates=number_of_classes_predicates,
                                           nof_objects=number_of_classes_objects,
                                           visual_features_predicate_size=self.VISUAL_FEATURES_PREDICATE_SIZE,
                                           visual_features_object_size=self.VISUAL_FEATURES_OBJECT_SIZE, is_train=False,
                                           rnn_steps=self.RNN_STEPS)
            # Create session
            self.session = tf.Session()
            # Restore variables from disk.
            # Initialize the Computational Graph
            init = tf.global_variables_initializer()
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            module_path = FilesManager().get_file_path("sg_module.train.saver")
            module_path_load = os.path.join(module_path, module_name + "_module.ckpt")
            if os.path.exists(module_path_load + ".index"):
                saver.restore(self.session, module_path_load)
            else:
                raise Exception("Module not found")

    def predict_rnn_belief_module(self, entity, using_gt_object_boxes=False):
        """
        Load module and run over list of entities
        :param entity: a single entity pre-processed by the Feature Extraction Module
        :param using_gt_object_boxes: A flag if we want to predict with GT object boxes or not.
        :return: an entity with updating predicates and objects probabilities
        """

        # create one hot vector for predicate_neg
        predicate_neg = np.zeros(self.rnn_belief_model.nof_predicates)
        predicate_neg[self.rnn_belief_model.nof_predicates - 1] = 1

        # get input place holders
        belief_predicate_ph, belief_object_ph, extended_belief_object_shape_ph, visual_features_predicate_ph, visual_features_object_ph = self.rnn_belief_model.get_in_ph()
        # get module output
        out_predicate_probes, out_object_probes = self.rnn_belief_model.get_output()

        # set diagonal to be neg
        indices = np.arange(entity.predicates_probes.shape[0])
        entity.predicates_outputs_with_no_activation[indices, indices, :] = predicate_neg
        entity.predicates_labels[indices, indices, :] = predicate_neg

        # get shape of extended object to be used by the module
        extended_belief_object_shape = np.asarray(entity.predicates_probes.shape)
        extended_belief_object_shape[2] = self.rnn_belief_model.nof_objects

        # Prediction using GT object boxes
        if using_gt_object_boxes:
            in_object_belief = entity.objects_labels * 10
        else:
            in_object_belief = entity.objects_outputs_with_no_activations

        # create the feed dictionary
        feed_dict = {belief_predicate_ph: entity.predicates_outputs_with_no_activation,
                     belief_object_ph: in_object_belief,
                     extended_belief_object_shape_ph: extended_belief_object_shape}

        out_predicate_probes_val, out_object_probes_val = self.session.run([out_predicate_probes, out_object_probes], feed_dict=feed_dict)
        # set diag in order to take in statistic
        out_predicate_probes_val[indices, indices, :] = predicate_neg

        entity.predicates_probes = np.copy(out_predicate_probes_val)
        entity.objects_probs = np.copy(out_object_probes_val)

        return entity

    def get_model(self, number_of_classes, weight_path, activation="softmax", use_mask=False):
        """
        This function loads the model
        :param use_mask: Using mask means different ResNet50 model
        :param activation: softmax is the default, otherwise its none
        :param weight_path: model weights path
        :param number_of_classes: number of classes
        :return: model
        """

        if K.image_dim_ordering() == 'th':
            if use_mask:
                input_shape_img = (5, None, None)
            else:
                input_shape_img = (3, None, None)
        else:
            if use_mask:
                input_shape_img = (self.config.crop_height, self.config.crop_width, 5)
            else:
                input_shape_img = (self.config.crop_height, self.config.crop_width, 3)

        img_input = Input(shape=input_shape_img, name="image_input")

        # Define ResNet50 model Without Top
        net = ModelZoo()
        if use_mask:
            model_resnet50 = net.resnet50_with_masking_dual(img_input, trainable=True)
        else:
            model_resnet50 = net.resnet50_base(img_input, trainable=True)

        model_resnet50 = GlobalAveragePooling2D(name='global_avg_pool')(model_resnet50)
        output_resnet50 = Dense(number_of_classes, kernel_initializer="he_normal", activation=activation, name='fc')(
            model_resnet50)

        # Define the model
        model = Model(inputs=img_input, outputs=output_resnet50, name='resnet50')
        # In the summary, weights and layers from ResNet50 part will be hidden, but they will be fit during the training
        # model.summary()

        # Load pre-trained weights for ResNet50
        try:
            model.load_weights(weight_path, by_name=True)
        except Exception as e:
            print('Could not load pretrained model weights. Weights can be found at {} and {}'.format(
                'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
                'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
            ))
            raise Exception(e)

        print('Finished successfully loading Model')
        return model

    def predict_objects_for_module(self, entity, objects, url_data, hierarchy_mapping_objects):
        """
        This function predicts objects for module later - object_probes [n, 150], object_features [n, 2048], object_labels
        [n, 150], objects_beliefs [n, 150]
        :param objects: List of objects
        :param entity: Entity visual genome class
        :param url_data: a url data
        :param hierarchy_mapping_objects: hierarchy_mapping for objects
        :return:
        """

        ## Get probabilities
        # Create a data generator for VisualGenome for OBJECTS
        data_gen_val_objects_vg = visual_genome_data_cnn_generator_with_batch(data=objects,
                                                                              hierarchy_mapping=hierarchy_mapping_objects,
                                                                              config=self.config, mode='validation',
                                                                              batch_size=self.NUM_BATCHES,
                                                                              evaluate=False)
        # Get the object probabilities [len(objects), 150]
        objects_probes = self.objects_model.predict_generator(data_gen_val_objects_vg,
                                                              steps=int(
                                                                  math.ceil(len(objects) / float(self.NUM_BATCHES))),
                                                              max_q_size=1, workers=1)
        # Save probabilities
        entity.objects_probs = np.copy(objects_probes)
        # del objects_probes

        ## Get GT labels
        # Get the GT labels - [len(objects), ]
        index_labels_per_gt_sample = np.array([hierarchy_mapping_objects[object.names[0]] for object in objects])
        # Get the max argument from the network output - [len(objects), ]
        index_labels_per_sample = np.argmax(objects_probes, axis=1)

        print("The Total number of Objects is {0} and {1} of them are positives".format(
            len(objects),
            np.where(index_labels_per_gt_sample == index_labels_per_sample)[0].shape[0]))
        print("The Objects accuracy is {0}".format(
            np.where(index_labels_per_gt_sample == index_labels_per_sample)[0].shape[0] / float(len(objects))))

        # Get the object labels on hot vector per object [len(objects), 150]
        objects_labels = np.eye(len(hierarchy_mapping_objects), dtype='uint8')[index_labels_per_gt_sample.reshape(-1)]
        # Save labels
        entity.objects_labels = np.copy(objects_labels)

        ## Get object features
        resized_img_lst = []
        # Define the function
        for object in objects:
            try:
                img = get_img(url_data)
                # Get the mask: a dict with {x1,x2,y1,y2}
                mask_object = get_mask_from_object(object)
                # Saves as a box
                object_box = np.array([mask_object['x1'], mask_object['y1'], mask_object['x2'], mask_object['y2']])
                patch = img[object_box[BOX.Y1]: object_box[BOX.Y2], object_box[BOX.X1]: object_box[BOX.X2], :]
                resized_img = get_img_resize(patch, self.config.crop_width, self.config.crop_height,
                                             type=self.config.padding_method)
                resized_img = np.expand_dims(resized_img, axis=0)
                resized_img_lst.append(resized_img)
            except Exception as e:
                print("Exception for object: {0}, image: {1}".format(object, url_data))
                print(str(e))
                traceback.print_exc()

        resized_img_arr = np.concatenate(resized_img_lst)
        size = len(resized_img_lst)

        # We are predicting in one forward pass 128*3 images
        batch_size = self.NUM_BATCHES

        if size % batch_size == 0:
            num_of_batches_per_epoch = size / batch_size
        else:
            num_of_batches_per_epoch = size / batch_size + 1

        objects_outputs_without_softmax = []
        for batch in range(num_of_batches_per_epoch):
            ## Get the object features [len(objects), 150]
            objects_noactivation_outputs = \
                self.noactivation_outputs_object_func([resized_img_arr[batch * batch_size: (batch + 1) * batch_size]])[
                    0]
            objects_outputs_without_softmax.append(objects_noactivation_outputs)

        # Save objects output with no activation (no softmax) - [len(objects), 150]
        entity.objects_outputs_with_no_activations = np.copy(np.concatenate(objects_outputs_without_softmax))

    def predict_predicates_for_module(self, entity, objects, url_data, hierarchy_mapping_predicates, use_mask=False):
        """
        This function predicts predicates for module later - predicates_probes [n, n, 51], predicates_features [n, n, 2048],
        predicates_labels [n, n, 51], predicate_beliefs [n, n, 51]
        :param use_mask: Using mask means different ResNet50 model and different generator
        :param objects: List of objects
        :param entity: Entity visual genome class
        :param url_data: a url data
        :param hierarchy_mapping_predicates: hierarchy_mapping for predicates
        :return:
        """

        # Create object pairs
        objects_pairs = list(itertools.product(objects, repeat=2))

        # Create a dict with key as pairs - (subject, object) and their values are predicates use for labels
        relations_dict = {}

        # Create a dict with key as pairs - (subject, object) and their values are relation index_id
        relations_filtered_id_dict = {}
        for relation in entity.relationships:
            relations_dict[(relation.subject.id, relation.object.id)] = relation.predicate
            relations_filtered_id_dict[(relation.subject.id, relation.object.id)] = relation.filtered_id

        # Create a data generator for VisualGenome for PREDICATES depends using masks
        if use_mask:
            data_gen_val_predicates_vg = visual_genome_data_predicate_mask_dual_pairs_generator_with_batch(
                data=objects_pairs,
                relations_dict=relations_dict,
                hierarchy_mapping=hierarchy_mapping_predicates,
                config=self.config,
                mode='validation',
                batch_size=self.NUM_BATCHES,
                evaluate=False)
        else:
            data_gen_val_predicates_vg = visual_genome_data_predicate_pairs_generator_with_batch(data=objects_pairs,
                                                                                                 relations_dict=relations_dict,
                                                                                                 hierarchy_mapping=hierarchy_mapping_predicates,
                                                                                                 config=self.config,
                                                                                                 mode='validation',
                                                                                                 batch_size=self.NUM_BATCHES,
                                                                                                 evaluate=False)

        ## Get the Predicate probabilities [n, 51]
        predicates_probes = self.predicates_model.predict_generator(data_gen_val_predicates_vg,
                                                                    steps=int(math.ceil(
                                                                        len(objects_pairs) / float(self.NUM_BATCHES))),
                                                                    max_q_size=1, workers=1)
        # Reshape the predicates probabilites [n, n, 51]
        reshaped_predicates_probes = predicates_probes.reshape(
            (len(objects), len(objects), len(hierarchy_mapping_predicates)))
        # Save probabilities
        entity.predicates_probes = np.copy(reshaped_predicates_probes)
        # del predicates_probes

        ## Get labels
        # Get the GT mapping labels - [ len(objects_pairs), ]
        index_labels_per_gt_sample = np.array(
            [hierarchy_mapping_predicates[relations_dict[(pair[0].id, pair[1].id)]]
             if (pair[0].id, pair[1].id) in relations_dict else hierarchy_mapping_predicates['neg']
             for pair in objects_pairs])

        ## Get the predicate GT label by name - [ len(objects_pairs), ]
        predicate_gt_sample = np.array(
            [relations_dict[(pair[0].id, pair[1].id)] if (pair[0].id, pair[1].id) in relations_dict else -1
             for pair in objects_pairs])

        reshape_predicate_gt_sample = predicate_gt_sample.reshape(len(objects), len(objects))
        entity.predicates_gt_names = np.copy(reshape_predicate_gt_sample)

        ## Get the predicate GT label by id - [ len(objects_pairs), ]
        relation_filtered_id_gt_sample = np.array([relations_filtered_id_dict[(pair[0].id, pair[1].id)]
                                                   if (pair[0].id, pair[1].id) in relations_filtered_id_dict else -1
                                                   for pair in objects_pairs])

        reshape_relation_filtered_id_gt_sample = relation_filtered_id_gt_sample.reshape(len(objects), len(objects))
        entity.predicates_relation_filtered_id = np.copy(reshape_relation_filtered_id_gt_sample)

        # Get the max argument - [len(objects_pairs), ]
        index_labels_per_sample = np.argmax(predicates_probes, axis=1)

        # Check how many positives and negatives relation we have
        pos_indices = []
        id = -1
        for pair in objects_pairs:
            id += 1
            sub = pair[0]
            obj = pair[1]
            if (sub.id, obj.id) in relations_dict and relations_dict[(sub.id, obj.id)] != "neg":
                pos_indices.append(id)

        print("The Total number of Relations is {0} while {1} of them positives and {2} of them negatives ".
              format(len(objects_pairs), len(pos_indices), len(objects_pairs) - len(pos_indices)))

        print("The Total Relations accuracy is {0}".format(
            np.where(index_labels_per_gt_sample == index_labels_per_sample)[0].shape[0] / float(len(objects_pairs))))

        # Check for no divide by zero because we don't have any *POSITIVE* relations
        if np.sum(index_labels_per_gt_sample != hierarchy_mapping_predicates['neg']) == 0:
            print("The Positive Relations accuracy is 0 - We have no positive relations")
        else:
            print("The Positive Relations accuracy is {0}".format(
                np.where((index_labels_per_gt_sample == index_labels_per_sample) &
                         (index_labels_per_gt_sample != hierarchy_mapping_predicates['neg']))[0].shape[0] /
                float(np.sum(index_labels_per_gt_sample != hierarchy_mapping_predicates['neg']))))

        # Check for no divide by zero because we don't have any *NEGATIVE* relations
        if np.sum(index_labels_per_gt_sample == hierarchy_mapping_predicates['neg']) == 0:
            print("The Negative Relations accuracy is 0 - We have no negative relations")
        else:
            print("The Negative Relations accuracy is {0}".format(
                np.where((index_labels_per_gt_sample == index_labels_per_sample) &
                         (index_labels_per_gt_sample == hierarchy_mapping_predicates['neg']))[0].shape[0] /
                float(np.sum(index_labels_per_gt_sample == hierarchy_mapping_predicates['neg']))))

        ## Get the object labels on hot vector per object [len(objects), 51]
        predicates_labels = np.eye(len(hierarchy_mapping_predicates), dtype='uint8')[
            index_labels_per_gt_sample.reshape(-1)]
        # Reshape the predicates labels [n, n, 51]
        reshaped_predicates_labels = predicates_labels.reshape(
            (len(objects), len(objects), len(hierarchy_mapping_predicates)))
        # Save labels
        entity.predicates_labels = np.copy(reshaped_predicates_labels)

        ## Get predicates features
        resized_img_lst = []
        # Define the function
        for object_pair in objects_pairs:
            try:
                # Get Image
                img = get_img(url_data)
                # Get Subject and Object
                subject = object_pair[0]
                object = object_pair[1]
                # Calc Union-Box
                # Get the Subject mask: a dict with {x1,x2,y1,y2}
                mask_subject = get_mask_from_object(subject)
                # Saves as a box
                subject_box = np.array([mask_subject['x1'], mask_subject['y1'], mask_subject['x2'], mask_subject['y2']])

                # Get the Object mask: a dict with {x1,x2,y1,y2}
                mask_object = get_mask_from_object(object)
                # Saves as a box
                object_box = np.array([mask_object['x1'], mask_object['y1'], mask_object['x2'], mask_object['y2']])

                # Get the UNION box: a BOX (numpy array) with [x1,x2,y1,y2]
                union_box = find_union_box(subject_box, object_box)

                patch = img[union_box[BOX.Y1]: union_box[BOX.Y2], union_box[BOX.X1]: union_box[BOX.X2], :]
                resized_img = get_img_resize(patch, self.config.crop_width, self.config.crop_height,
                                             type=self.config.padding_method)

                # Using mask dual means the data should be prepared
                if use_mask:
                    # Fill HeatMap
                    heat_map_subject = np.zeros(img.shape)
                    heat_map_subject[subject_box[BOX.Y1]: subject_box[BOX.Y2], subject_box[BOX.X1]: subject_box[BOX.X2],
                    :] = 255
                    heat_map_object = np.zeros(img.shape)
                    heat_map_object[object_box[BOX.Y1]: object_box[BOX.Y2], object_box[BOX.X1]: object_box[BOX.X2],
                    :] = 255

                    # Cropping the patch from the heat map.
                    patch_heatmap_heat_map_subject = heat_map_subject[union_box[BOX.Y1]: union_box[BOX.Y2],
                                                     union_box[BOX.X1]: union_box[BOX.X2], :]
                    patch_heatmap_heat_map_object = heat_map_object[union_box[BOX.Y1]: union_box[BOX.Y2],
                                                    union_box[BOX.X1]: union_box[BOX.X2], :]

                    # Resize the image according the padding method
                    resized_heatmap_subject = get_img_resize(patch_heatmap_heat_map_subject, self.config.crop_width,
                                                             self.config.crop_height, type=self.config.padding_method)
                    resized_heatmap_object = get_img_resize(patch_heatmap_heat_map_object, self.config.crop_width,
                                                            self.config.crop_height, type=self.config.padding_method)

                    # Concatenate the heat-map to the image in the kernel axis
                    resized_img = np.concatenate((resized_img, resized_heatmap_subject[:, :, :1]), axis=2)
                    resized_img = np.concatenate((resized_img, resized_heatmap_object[:, :, :1]), axis=2)

                    # endregion

                resized_img = np.expand_dims(resized_img, axis=0)
                resized_img_lst.append(resized_img)

            except Exception as e:
                print("Exception for object: {0}, image: {1}".format(object_pair, url_data))
                print(str(e))
                traceback.print_exc()

        resized_img_arr = np.concatenate(resized_img_lst)

        size = len(resized_img_lst)
        # We are predicting in one forward pass 128*3 images
        # batch_size = NUM_BATCHES * 3
        batch_size = self.NUM_BATCHES

        if size % batch_size == 0:
            num_of_batches_per_epoch = size / batch_size
        else:
            num_of_batches_per_epoch = size / batch_size + 1

        predicates_outputs_without_softmax = []
        for batch in range(num_of_batches_per_epoch):
            ##  Get the predicate beliefs [len(objects), len(objects), 51]
            predict_noactivation_outputs = \
                self.noactivation_outputs_predicate_func(
                    [resized_img_arr[batch * batch_size: (batch + 1) * batch_size]])[0]
            predicates_outputs_without_softmax.append(predict_noactivation_outputs)

        # Concatenate to [n*n, 51]
        predicates_outputs_with_no_activation = np.concatenate(predicates_outputs_without_softmax)
        # Number of features
        number_of_outputs = predicates_outputs_with_no_activation.shape[1]
        # Reshape the predicates labels [n, n, 51]
        reshaped_predicates_outputs_with_no_activation = predicates_outputs_with_no_activation.reshape((len(objects),
                                                                                                        len(objects),
                                                                                                        number_of_outputs))
        # Save predicate outputs with no activations (no softmax)
        entity.predicates_outputs_with_no_activation = np.copy(reshaped_predicates_outputs_with_no_activation)
