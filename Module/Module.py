import numpy as np

from LangModule import LangModule
from VisualModuleLazy import VisualModule
import cPickle


class Module(object):
    """
    Module for scene grapher
    This module includes visual module and language module
    """

    def __init__(self, object_ids, predicate_ids, lang_embed_size, visual_embed_size, objects_training_dir_name="",
                 predicates_training_dir_name=""):
        """
        Initialize module and create module parameters
        :param nof_objects: number of object classes
        :param nof_predicates: number of predicate classes
        :param lang_embed_size: size of embedded word in word2vec space
        :param visual_embed_size: size of features extracted from predicate CNN
        :param objects_training_dir_name: objects training dir name for taking the weights
        :param predicates_training_dir_name: predicates training dir name for taking the weights
        """
        # save input params
        self.nof_objects = len(object_ids)
        self.object_ids = object_ids
        self.nof_predicates = len(predicate_ids)
        self.predicate_ids = predicate_ids
        self.lang_embed_size = lang_embed_size
        self.visual_embed_size = visual_embed_size

        # create language module
        self.lang = LangModule(object_ids, predicate_ids)

        # create visual module
        self.visual = VisualModule(predicates_training_dir_name=predicates_training_dir_name,
                                   objects_training_dir_name=objects_training_dir_name)

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
        # s = np.random.randn(*self.s_dimensions)
        # load init paramters to be equal to predicate CNN	
        file_handle = open("last_layer_weights.p", "rb")
        z = cPickle.load(file_handle).T
        file_handle.close()
        # w = np.zeros(self.w_dimensions)
        # b = np.zeros(self.b_dimensions)
        # z = np.zeros(self.z_dimensions)
        s = np.zeros(self.s_dimensions)

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

    def get_gradient_and_loss(self, params, R1, R2, coeff_l=0.0, coeff_k=0.0, coeff_reg_visual=0.001):
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
        grad_z_reg = 2 * z
        grad_s_reg = 2 * s
        REG_VIS = np.sum(np.multiply(z, z)) + np.sum(np.multiply(s, s))

        ### total loss and grad
        loss = coeff_k * K + coeff_l * L + C + coeff_reg_visual * REG_VIS
        grad_w = coeff_k * grad_w_k + coeff_l * grad_w_l + grad_w_c
        grad_b = coeff_k * grad_b_k + coeff_l * grad_b_l + grad_b_c
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
        # decode module parmas
        w, b, z, s = self.decode_parameters(params)

        # get tensor of probabilities (element per any relation - triplet)
        lang_predict = self.lang.predict_all(w, b)

        images_score = 0
        for img in images:
            k_highest_confidence = []
            min_predict_confidence = 0
            # extract features from visual module and probabilities for subject, object and predicate

            # iterate over each relation to predict and find k highest predictions
            predictions = []
            for subject_index in range(len(img.obects)):
                for object_index in range(len(img.obects)):
                    # filter if subject equals to object
                    if object_index == subject_index:
                        continue
                    # extract features and probabilities
                    predicate_features, subject_prob, object_prob = self.visual.extract_features_for_evaluate \
                        (img.obects[subject_index], img.obects[object_index], img.image.url)
                    predicate_prob = self.visual.predicate_predict(predicate_features, z, s)

                    # calc tensor of probabilities of visual moudle
                    visual_predict = np.multiply.outer(subject_prob, np.multiply.outer(predicate_prob, object_prob))

                    # calc tensor of probabilities taking into account both visual and language module
                    predict_tensor = visual_predict * lang_predict

                    # get the highset probabilities
                    predict = np.argmax(predict_tensor)
                    # convert to relation indexes (triplet)
                    predict_triplet = np.unravel_index(predict, predict_tensor.shape)
                    predict_confidence = predict_tensor[predict_triplet]
                    while predict_confidence > min_predict_confidence:
                        # append to predictions list
                        predictions.append((subject_index, object_index), predict_triplet,
                                           predict_tensor[predict_triplet])
                        # remove confidence from tensor
                        predict_tensor[predict_triplet] = 0
                        # remove lowest confidence
                        if len(predictions) == k:
                            predictions_arr = np.asarray(predictions)
                            index_to_remove = predictions_arr[:, 2].argmin()
                            predictions.remove(predictions[index_to_remove])

            # FIXME: calc how many of the ground truth relationships included in k highest confidence relationships
            images_score += 0

        return images_score / len(images)
