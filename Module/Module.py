from LangModule import LangModule
from VisualModule import VisualModule
import numpy as np


class Module(object):
    """
    Module for scene grapher
    This module includes visual module and language module
    """

    def __init__(self, nof_objects, nof_predicates, lang_embed_size, visual_embed_size):
        """
        Initialize module and create module parameters
        :param nof_objects: number of object classes
        :param nof_predicates: number of predicate classes
        :param lang_embed_size: size of embedded word in word2vec space
        :param visual_embed_size: size of features extracted from predicate CNN
        """
        # save input params
        self.nof_objects = nof_objects
        self.nof_predicates = nof_predicates
        self.lang_embed_size = lang_embed_size
        self.visual_embed_size = visual_embed_size

        # create language module
        self.lang = LangModule()

        # create visual module
        self.visual = VisualModule()

        # create dimensions for module parameters
        self.w_dimensions = (nof_predicates, 2 * lang_embed_size)
        self.b_dimensions = (nof_predicates, 1)
        self.z_dimensions = (nof_predicates, visual_embed_size)
        self.s_dimensions = (nof_predicates, 1)
        self.dimensions = [self.w_dimensions, self.b_dimensions, self.z_dimensions, self.s_dimensions]

        # create parameters
        w = np.random.randn(*self.w_dimensions)
        b = np.random.randn(*self.b_dimensions)
        z = np.random.randn(*self.z_dimensions)
        s = np.random.randn(*self.s_dimensions)

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

    def get_gradient_and_loss(self, params, R1, R2, coeff_l=0.05, coeff_k=0.002):
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
        r2_f = self.lang.likelihood(r2_embed, w, b, R2.predicate_ids)

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
        l_coeff[R1.instances < R2.instances] = -1 * coeff_l
        l_coeff[R1.instances == R2.instances] = 0

        #
        # find max of f(Rmin) - f(Rmax) + 1
        #
        l_loss = np.maximum(np.multiply(l_coeff, f_sub) + 1, 0)
        L = np.sum(l_loss) / l_loss.shape[0]

        # grad W
        grad_w_l = np.zeros(w.shape)
        l_r1_coeffs = np.copy(l_loss)
        l_r1_coeffs[l_r1_coeffs != 0] = 1.0 / l_loss.shape[0]
        l_r1_coeffs = np.multiply(l_r1_coeffs, -1 * l_coeff)
        np.add.at(grad_w_l, R1.predicate_ids, l_r1_coeffs * r1_embed)
        l_r2_coeffs = np.copy(l_loss)
        l_r2_coeffs[l_r2_coeffs != 0] = 1.0 / l_loss.shape[0]
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
        predicate_features, subject_probabilities, object_probabilities = self.visual.extract_features(R1)
        for index in range(len(R1.worda)):
            r_v = self.visual.likelihood(R1.subject_ids[index], R1.object_ids[index], R1.predicate_ids[index],
                                         predicate_features[index], subject_probabilities[index],
                                         object_probabilities[index], z, s)
            r_likelihood = r_v * r1_f[index]

            r2_v = self.visual.likelihood(R2.subject_ids, R2.object_ids, R2.predicate_ids,
                                          predicate_features[index], subject_probabilities[index],
                                          object_probabilities[index], z, s)

            r2_likelihood = r2_v * r2_f.flatten()

            # filter relationship that identical to r1
            r2_filter = np.logical_and(
                np.logical_and(R2.worda == R1.worda[index], R2.predicate_ids == R1.predicate_ids[index]),
                R2.wordb == R1.wordb[index])
            r2_max_index = np.argmax(np.multiply(r2_likelihood, r2_filter.astype(int)))

            # loss
            c_loss = max(1 + r2_likelihood[r2_max_index] - r_likelihood, 0)
            C += c_loss / len(R1.worda)
            # gradients
            if c_loss != 0:
                # w gradient
                grad_w_c[R2.predicate_ids[r2_max_index]] += r2_v[r2_max_index] * r2_embed[r2_max_index]
                grad_w_c[R1.predicate_ids[index]] -= r_v * r1_embed[index]

                # b gradient
                grad_b_c[R2.predicate_ids[r2_max_index]] += r2_v[r2_max_index]
                grad_b_c[R1.predicate_ids[index]] -= r_v

                # z gradient
                grad_z_c[R2.predicate_ids[r2_max_index]] += predicate_features[index] * r2_f[r2_max_index]
                grad_z_c[R1.predicate_ids[index]] -= predicate_features[index] * r1_f[index]

                # s gradient
                grad_s_c[R2.predicate_ids[r2_max_index]] += r2_f[r2_max_index]
                grad_s_c[R1.predicate_ids[index]] -= r1_f[index]

        ### total loss and grad
        loss = coeff_k * K + coeff_l * L + C
        if loss < 0:
            print "debug"
        grad_w = coeff_k * grad_w_k + coeff_l * grad_w_l + grad_w_c
        grad_b = coeff_k * grad_b_k + coeff_l * grad_b_l
        grad_z = grad_z_c
        grad_s = grad_s_c

        grad = self.encode_parameters(grad_w, grad_b, grad_z, grad_s)

        return loss, grad

    def predict(self, worda, wordb, params):
        w, b, z, s = self.decode_parameters(params)

        return self.lang.predict(worda, wordb, w, b)
