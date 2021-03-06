import cPickle
import gc

import numpy as np

from LangModule import LangModule
from Utils.Logger import Logger
from Utils.ModuleDetection import ModuleDetection
from Utils.Utils import softmax_multi_dim
from VisualModule import VisualModule

class Module(object):
    """
    Module for scene grapher
    This module includes visual module and language module
    """

    def __init__(self, object_ids, predicate_ids, lang_embed_size, visual_embed_size):
        """
        Initialize module and create module parameters
        :param nof_objects: number of object classes
        :param nof_predicates: number of predicate classes
        :param lang_embed_size: size of embedded word in word2vec space
        :param visual_embed_size: size of features extracted from predicate CNN
        """
        # save input params
        self.nof_objects = len(object_ids)
        self.object_ids = object_ids
        self.reverse_object_ids = {self.object_ids[id] : id for id in self.object_ids}
        self.nof_predicates = len(predicate_ids)
        self.predicate_ids = predicate_ids
        self.reverse_predicate_ids = {self.predicate_ids[id] : id for id in self.predicate_ids}
        self.lang_embed_size = lang_embed_size
        self.visual_embed_size = visual_embed_size

        # create language module
        self.lang = LangModule(object_ids, predicate_ids, lang_embed_size)

        # create visual module
        self.visual = VisualModule()

        # create dimensions for module parameters
        self.w_dimensions = (self.nof_predicates, 2 * lang_embed_size)
        self.b_dimensions = (self.nof_predicates, 1)
        self.z_dimensions = (self.nof_predicates, visual_embed_size)
        self.s_dimensions = (self.nof_predicates, 1)
        self.dimensions = [self.w_dimensions, self.b_dimensions, self.z_dimensions, self.s_dimensions]

        # create parameters
        w = np.random.randn(*self.w_dimensions)
        b = np.random.randn(*self.b_dimensions)
        z = np.random.randn(*self.z_dimensions)
        s = np.random.randn(*self.s_dimensions)
        # load init paramters to be equal to predicate CNN	
        file_handle = open("last_layer_weights.p", "rb")
        z = cPickle.load(file_handle).T
        file_handle.close()
        # w = np.zeros(self.w_dimensions)
        # b = np.zeros(self.b_dimensions)
        # z = np.zeros(self.z_dimensions)
        # s = np.zeros(self.s_dimensions)

        # encode parameters
        self.params = self.encode_parameters(w, b, z, s)

    def encode_parameters(self, w, b, z, s):
        """
        Encode module parameters to one array
        :param w: language param
        :param b: language param
        :param z: visual param
        :param s: visual param
        :return: one array of all the parameters
        """
        return np.concatenate((w.flatten(), b.flatten(), z.flatten(), s.flatten()))

    def decode_parameters(self, parameters):
        """
        Decode array of parameters to module paramters
        :param parameters: array of parameters
        :return: meaningful parameters w, b, z, s
        """
        nof_w_parameters = self.w_dimensions[0] * self.w_dimensions[1]
        w = np.reshape(parameters[:nof_w_parameters], self.w_dimensions)
        rest_params = parameters[nof_w_parameters:]

        nof_b_parameters = self.b_dimensions[0] * self.b_dimensions[1]
        b = np.reshape(rest_params[:nof_b_parameters], self.b_dimensions)
        rest_params = rest_params[nof_b_parameters:]

        nof_z_parameters = self.z_dimensions[0] * self.z_dimensions[1]
        z = np.reshape(rest_params[:nof_z_parameters], self.z_dimensions)
        rest_params = rest_params[nof_z_parameters:]

        nof_s_parameters = self.s_dimensions[0] * self.s_dimensions[1]
        s = np.reshape(rest_params[:nof_s_parameters], self.s_dimensions)

        return w, b, z, s

    def get_params(self):
        """
        Get module  parameters
        :return: module parameters
        """
        return self.params

    def set_weights(self, params):
        """
        Set module parameters
        :param params:
        :return: None
        """
        self.params = params

    def get_gradient_and_loss(self, params, R1, R2, coeff_l=0.0, coeff_k=0.0, coeff_reg_visual=0.001, coeff_reg_lang=0.001):
        """
        Calculate the cost and the gradient with respect to model parameters

        Loss(R) = coeff_k * K(R) + coeff_l * L(R) + C(R)
                                                                                            |f(Ri) - f(Rj)|
        K(R) is the variance of projection function - K(R) = variance(Dij) =  variance({------------------------})
                                                                                         cosine_distance(Ri, Rj)

        L(R) is the likelihood of relationship - L(R) = Sum-ij(max{f(Ri) - f(Rj) + 1, 0})
                                                    Rj occurs more frequently than Ri
        :param coeff_k: coefficient of K
        :param coeff_l: coefficient of L
        :param params: module parameters
        :param R1: Data Object represents group of relationships to compare to R2 group
        :param R2: Data Object represents group of relationships to compare to R1 group
        :return: loss and gradient for each parameter in x

        """
        w, b, z, s = self.decode_parameters(params)

        # get language embedding
        r1_a_embed = self.lang.word_embed(R1.worda)
        r1_b_embed = self.lang.word_embed(R1.wordb)
        r1_pred_embed = self.lang.word_embed(R1.predicate)
        r1_embed = self.lang.relation_embed(r1_a_embed, r1_b_embed)
        r2_a_embed = self.lang.word_embed(R2.worda)
        r2_b_embed = self.lang.word_embed(R2.wordb)
        r2_pred_embed = self.lang.word_embed(R2.predicate)
        r2_embed = self.lang.relation_embed(r2_a_embed, r2_b_embed)

        # get language likelihood
        r1_f = self.lang.likelihood(r1_embed, w, b, R1.predicate_ids)
        # r1_f = np.ones(r1_f.shape)
        r2_f = self.lang.likelihood(r2_embed, w, b, R2.predicate_ids)
        # r2_f = np.ones(r2_f.shape)

        # get distance in word2vec space
        dist = self.lang.distance(r1_a_embed, r1_b_embed, r1_pred_embed, r2_a_embed, r2_b_embed, r2_pred_embed)

        ### K loss and gradient
        # calc D
        #               |f(Ri) - f(Rj)|^2
        #       D = ------------------------
        #             cosine_distance(Ri, Rj)
        #
        f_sub = np.subtract(r2_f, r1_f)
        D = np.divide(f_sub, dist.reshape(-1, 1))
        D = np.multiply(D, f_sub)
        avg_d = np.average(D)
        # calc K - variance of D
        K = np.var(D)

        # calc gradient of K with respect to W
        #
        # dk          4 *  (D - Avg(D))
        # --  =  ------------------------------ (f(Ri) - f(Rj)) * Ri    <------- Ri predicate is k
        # dwk      M * cosine_distance(Ri, Rj)
        #
        #
        # dk        4  * (D - Avg(D))
        # --  = - ------------------------------- (f(Ri) - f(Rj)) * Rj    <------- Rj predicate is k
        # dwk       M * (cosine_distance(Ri, Rj))
        #
        # or zero otherwise
        #
        grad_w_coeff = 4 * np.divide(np.subtract(D, avg_d), dist.reshape(-1, 1)) / D.shape[0]
        grad_w_coeff = np.multiply(grad_w_coeff, f_sub)
        grad_w_k = np.zeros(w.shape)
        np.add.at(grad_w_k, R1.predicate_ids, - grad_w_coeff * r1_embed)
        np.add.at(grad_w_k, R2.predicate_ids, grad_w_coeff * r2_embed)

        # calc K gradient B - nothing to do (always zero)
        grad_b_k = np.zeros(b.shape).reshape(-1, 1)
        np.add.at(grad_b_k, R1.predicate_ids, - grad_w_coeff)
        np.add.at(grad_b_k, R2.predicate_ids, grad_w_coeff)

        ### L loss and gradient
        #
        # l_coeff - 1 if R1 occurs more then R2, -1 of R2 occurs more then R1 and 0 otherwise.
        #
        l_coeff = np.ones(R1.instances.shape).reshape(-1, 1)
        l_coeff[R1.instances < R2.instances] = -1
        l_coeff[R1.instances == R2.instances] = 0

        #
        # find max of f(Rmin) - f(Rmax) + 1
        #
        l_loss = np.maximum(np.multiply(l_coeff, f_sub) + 1, 0)
        L = np.sum(l_loss) / l_loss.shape[0]

        # grad W
        grad_w_l = np.zeros(w.shape)
        # l_r1_coeffs will be 1/batch_size for every the likelihood of the less popular r is higher
        l_r1_coeffs = np.copy(l_loss)
        l_r1_coeffs[l_r1_coeffs != 0] = 1.0 / l_loss.shape[0]
        # if R1 is the less popular multiply by -1
        l_r1_coeffs = np.multiply(l_r1_coeffs, -1 * l_coeff)

        np.add.at(grad_w_l, R1.predicate_ids, l_r1_coeffs * r1_embed)
        # l_r1_coeffs will be 1/batch_size for every the likelihood of the less popular r is higher
        l_r2_coeffs = np.copy(l_loss)
        l_r2_coeffs[l_r2_coeffs != 0] = 1.0 / l_loss.shape[0]
        # if R2 is the less popular multiply by -1
        l_r2_coeffs = np.multiply(l_r2_coeffs, l_coeff)
        np.add.at(grad_w_l, R2.predicate_ids, l_r2_coeffs * r2_embed)

        # L grad b
        grad_b_l = np.zeros(b.shape).reshape(-1, 1)
        np.add.at(grad_b_l, R1.predicate_ids, l_r1_coeffs)
        np.add.at(grad_b_l, R2.predicate_ids, l_r2_coeffs)

        ### C loss and gradient
        grad_w_c = np.zeros(w.shape)
        grad_b_c = np.zeros(b.shape)
        grad_z_c = np.zeros(z.shape)
        grad_s_c = np.zeros(s.shape)
        C = 0
        predicate_features, subject_probabilities, object_probabilities = self.visual.extract_features(R1.relation_ids)
        predicate_prob = self.visual.predicate_predict(predicate_features, z, s)
        # get tensor of probabilities (element per any relation - triplet)
        lang_predict = self.lang.predict_all(w, b)
        for index in range(len(R1.worda)):

            # calc tensor of probabilities of visual moudle
            visual_predict = np.multiply.outer(subject_probabilities[index],
                                               np.multiply.outer(predicate_prob[index], object_probabilities[index]))
            # calc tensor of probabilities taking into account both visual and language module
            predict_tensor = visual_predict * lang_predict
            # copy and delete the true relation and find the next mask
            r_likelihood = predict_tensor[R1.subject_ids[index]][R1.predicate_ids[index]][R1.object_ids[index]]
            predict_tensor[R1.subject_ids[index]][R1.predicate_ids[index]][R1.object_ids[index]] = 0
            # get the highest probability
            predict = np.argmax(predict_tensor)
            # convert to relation indexes (triplet)
            predict_triplet = np.unravel_index(predict, predict_tensor.shape)

            max_likelihood = predict_tensor[predict_triplet]

            # loss
            c_loss = max(1 + max_likelihood - r_likelihood, 0)
            C += c_loss / len(R1.worda)
            # gradients
            if c_loss != 0:
                # get parameters for the gradient
                max_sub_prob = subject_probabilities[index][predict_triplet[0]]
                max_obj_prob = object_probabilities[index][predict_triplet[2]]
                true_sub_prob = subject_probabilities[index][R1.subject_ids[index]]
                true_obj_prob = object_probabilities[index][R1.object_ids[index]]
                max_f = lang_predict[predict_triplet]
                true_f = r1_f[index]
                max_v = visual_predict[predict_triplet]
                true_v = visual_predict[R1.subject_ids[index]][R1.predicate_ids[index]][R1.object_ids[index]]
                max_lang_features = self.lang.get_relation_embed(predict_triplet[0], predict_triplet[2])
                true_lang_features = r1_embed[index]
                visual_features = predicate_features[index]

                # w gradient
                grad_w_c[predict_triplet[1]] += max_v * max_lang_features / len(R1.worda)
                grad_w_c[R1.predicate_ids[index]] -= true_v * true_lang_features / len(R1.worda)

                # b gradient
                grad_b_c[predict_triplet[1]] += max_v / len(R1.worda)
                grad_b_c[R1.predicate_ids[index]] -= true_v / len(R1.worda)

                # z gradient
                grad_z_c[predict_triplet[1]] += max_sub_prob * max_obj_prob * visual_features * max_f / len(R1.worda)
                grad_z_c[R1.predicate_ids[index]] -= true_sub_prob * true_obj_prob * visual_features * true_f / len(
                    R1.worda)

                # s gradient
                grad_s_c[predict_triplet[1]] += max_sub_prob * max_obj_prob * max_f / len(R1.worda)
                grad_s_c[R1.predicate_ids[index]] -= true_sub_prob * true_obj_prob * true_f / len(R1.worda)

        ### loss and gradient of regularization
        #   reg_visual = Sum(Z ** Z)
        #   gradient_z(reg_visual) = 2*Z
        #   gradient_s(reg_visual) = 2*S
        #   ...
        grad_w_reg = 2 * w
        grad_b_reg = 2 * b
        grad_z_reg = 2 * z
        grad_s_reg = 2 * s
        REG_VIS = np.sum(np.multiply(z, z)) + np.sum(np.multiply(s, s))
        REG_LANG = np.sum(np.multiply(w, w) + np.sum(np.multiply(b, b)))
        ### total loss and grad
        loss = coeff_k * K + coeff_l * L + C + coeff_reg_visual * REG_VIS + coeff_reg_lang * REG_LANG
        grad_w = coeff_k * grad_w_k + coeff_l * grad_w_l + grad_w_c + coeff_reg_lang * grad_w_reg
        grad_b = coeff_k * grad_b_k + coeff_l * grad_b_l + grad_b_c + coeff_reg_lang * grad_b_reg
        grad_z = grad_z_c + coeff_reg_visual * grad_z_reg
        # grad_z = np.zeros(grad_z.shape)
        grad_s = grad_s_c + coeff_reg_visual * grad_s_reg
        # grad_s = np.zeros(grad_s.shape)
        grad = self.encode_parameters(grad_w, grad_b, grad_z, grad_s)

        return loss, grad

    def predict(self, relations, params):
        """
        Predict a relation (triplet) given detection of the objects.
        :param relations: detection of the objects
        :param params: module params
        :return:
        """
        # decode module parmas
        w, b, z, s = self.decode_parameters(params)

        # extract features from visual module and probabilities for subject, object and predicate
        predicate_features, subject_prob, object_prob = self.visual.extract_features(relations.relation_ids)
        predicate_prob = self.visual.predicate_predict(predicate_features, z, s)
        # get tensor of probabilities (element per any relation - triplet)
        lang_predict = self.lang.predict_all(w, b)
        # lang_predict = np.ones(lang_predict.shape)

        # iterate over each relation to predict
        predictions = []
        accuracy_percent = []
        for index in range(len(relations.worda)):
            # calc tensor of probabilities of visual moudle
            visual_predict = np.multiply.outer(subject_prob[index],
                                               np.multiply.outer(predicate_prob[index], object_prob[index]))
            # calc tensor of probabilities taking into account both visual and language module
            predict_tensor = visual_predict * lang_predict
            # get the highset probability
            predict = np.argmax(predict_tensor)
            # convert to relation indexes (triplet)
            predict_triplet = np.unravel_index(predict, predict_tensor.shape)
            # append to predictions list
            predictions.append(predict_triplet)
            # get accuracy percent
            correct_percent = predict_tensor[relations.subject_ids[index]][relations.predicate_ids[index]][
                                  relations.object_ids[index]] / np.sum(predict_tensor)
            accuracy_percent.append(correct_percent)
        return predictions, accuracy_percent

    def r_k_metric(self, images, k, params):
        """
        R@K metric measures  the fraction of ground truth relationships triplets
        that appear among the top k most confident triplet in an image
        :param images: data per image including the ground truth
        :param params: module params
        :return: R@K meric
        """
        # get logger
        logger = Logger()

        # decode module parmas
        w, b, z, s = self.decode_parameters(params)

        # get tensor of probabilities (element per any relation - triplet)
        lang_predict = self.lang.predict_all(w, b)

        # scores
        total_gt_relationships = 0
        total_score = 0

        for img in images:
            # filter images with no relationships
            if len(img.relationships) == 0:
                continue

            # create module detections
            detections = ModuleDetection(img, self.reverse_object_ids, self.reverse_predicate_ids)

            # iterate over each relation to predict and find k highest predictions
            top_predictions = np.zeros((0,))
            top_likelihoods = np.zeros((0,))
            top_k_global_subject_ids = np.zeros((0,))
            top_k_global_object_ids = np.zeros((0,))
            for subject_index in range(len(img.objects)):
                for object_index in range(len(img.objects)):
            #for subject_index in [1]:
            #    for relation in img.relationships:

                    # filter if subject equals to object
                    if (subject_index == object_index):
                        continue

                    #subject = relation.subject
                    #object = relation.object

                    subject = img.objects[subject_index]
                    object = img.objects[object_index]

                    # extract features and probabilities
                    subject_prob, object_prob, predicate_features = self.visual.extract_features_for_evaluate \
                        (subject, object, img.image.url)

                    #predicate_features, subject_prob, object_prob = self.visual.extract_features([relation.filtered_id])
                    #predicate_features = predicate_features[0]; subject_prob = subject_prob[0]; object_prob = object_prob[0]

                    predicate_prob = self.visual.predicate_predict(predicate_features, z, s)

                    # calc tensor of probabilities of visual moudle
                    visual_predict = np.multiply.outer(subject_prob, np.multiply.outer(predicate_prob.flatten(), object_prob))

                    # calc tensor of probabilities taking into account both visual and language module
                    predict_tensor = visual_predict * lang_predict
                    predict_tensor = softmax_multi_dim(predict_tensor)
                    # remove negative probabilties
                    predict_tensor[:,self.predicate_ids["neg"] ,:] = 0
                    # get the highset probabilities
                    max_k_predictions = np.argsort(predict_tensor.flatten())[-k:]
                    max_k_predictions_triplets = np.unravel_index(max_k_predictions, predict_tensor.shape)
                    max_k_subjects = max_k_predictions_triplets[0]
                    max_k_predicates = max_k_predictions_triplets[1]
                    max_k_objects = max_k_predictions_triplets[2]
                    max_k_likelihoods = predict_tensor[max_k_subjects, max_k_predicates, max_k_objects]

                    # append to the list of highest predictions
                    top_predictions = np.concatenate((top_predictions, max_k_predictions))
                    top_likelihoods = np.concatenate((top_likelihoods, max_k_likelihoods))

                    # store the relevant subject and object
                    max_k_global_subject_ids = np.ones(max_k_likelihoods.shape) * subject.id
                    max_k_global_object_ids = np.ones(max_k_likelihoods.shape) * object.id
                    top_k_global_subject_ids = np.concatenate((top_k_global_subject_ids, max_k_global_subject_ids))
                    top_k_global_object_ids = np.concatenate((top_k_global_object_ids, max_k_global_object_ids))

            # get k highest confidence
            top_k_indices = np.argsort(top_likelihoods)[-k:]
            predictions = top_predictions[top_k_indices]
            global_sub_ids = top_k_global_subject_ids[top_k_indices]
            global_obj_ids = top_k_global_object_ids[top_k_indices]
            likelihoods = top_likelihoods[top_k_indices]
            triplets = np.unravel_index(predictions.astype(int), predict_tensor.shape)
            for i in range(k):
                detections.add_detection(global_subject_id=global_sub_ids[i], global_object_id=global_obj_ids[i], pred_subject=triplets[0][i], pred_object=triplets[2][i], pred_predicate=triplets[1][i], top_k_index=i, confidence=likelihoods[i])

            img_score = 0
            nof_pos_relationship = 0
            for relation in img.relationships:
                # filter negative relationship
                if relation.predicate == "neg":
                    continue
                nof_pos_relationship += 1

                sub_id = self.lang.object_ids[relation.subject.names[0]]
                obj_id = self.lang.object_ids[relation.object.names[0]]
                predicate_id = self.lang.predicate_ids[relation.predicate]
                gt_relation = np.ravel_multi_index((sub_id, predicate_id, obj_id), predict_tensor.shape)

                # filter the predictions for the specific subject
                sub_predictions_indices = set(np.where(global_sub_ids == relation.subject.id)[0])
                obj_predictions_indices = set(np.where(global_obj_ids == relation.object.id)[0])
                relation_indices = set(np.where(predictions == gt_relation)[0])

                indices = sub_predictions_indices & obj_predictions_indices & relation_indices
                if len(indices) != 0:
                    img_score += 1
            total_score += img_score
            total_gt_relationships += nof_pos_relationship
            score = float(total_score) / total_gt_relationships
            img_score_precent = float(img_score)/nof_pos_relationship
            logger.log("image score: " + str(img_score_precent))
            logger.log("total score: " + str(score))
            detections.save_stat(score=img_score_precent)
            gc.collect()

        #return images_score / len(images)


    def predicate_class_recall(self, images, params, k = 5):
        """

        :param images:
        :return:
        """
        # get logger
        logger = Logger()

        # decode module parmas
        w, b, z, s = self.decode_parameters(params)

        # get tensor of probabilities (element per any relation - triplet)
        lang_predict = self.lang.predict_all(w, b)

        correct = np.zeros(len(self.predicate_ids))
        total = np.zeros(len(self.predicate_ids))
        for img in images:
            # filter images with no relationships
            if len(img.relationships) == 0:
                continue

            for relation in img.relationships:
                predicate_features, subject_prob, object_prob = self.visual.extract_features([relation.filtered_id])
                predicate_features = predicate_features[0]; subject_prob = subject_prob[0]; object_prob = object_prob[0]

                predicate_prob = self.visual.predicate_predict(predicate_features, z, s)

                # calc tensor of probabilities taking into account both visual and language module
                subject_class = self.object_ids[relation.subject.names[0]]
                object_class = self.object_ids[relation.object.names[0]]
                predicate_class = self.predicate_ids[relation.predicate]
		alter_predicate_class = self.predicate_ids[relation.predicate]
                if relation.predicate.islower() and self.predicate_ids.has_key(relation.predicate.upper()):  
		    alter_predicate_class = self.predicate_ids[relation.predicate.upper()] 		
		if relation.predicate.isupper() and self.predicate_ids.has_key(relation.predicate.lower()):
                    alter_predicate_class = self.predicate_ids[relation.predicate.lower()]

                predict_tensor = predicate_prob * lang_predict[subject_class, :, object_class]

                # remove negative probabilties
                max_k_predictions = np.argsort(predict_tensor)[-k:]
                found = np.where(predicate_class == max_k_predictions)[0]
                found_alter = np.where(alter_predicate_class == max_k_predictions)[0]
                if len(found) != 0 or len(found_alter) != 0:
                    correct[predicate_class] += 1
                total[predicate_class] += 1

	for i in range(len(self.reverse_predicate_ids)):
    	    if total[i] != 0:
        	logger.log("{0} recall@5 is {1} (total - {2}, correct {3})".format(self.reverse_predicate_ids[i], float(correct[i])/total[i], total[i], correct[i]))


        

