import numpy as np
import cPickle
from WordEmbd import WordEmbd
from scipy import spatial
from numpy.core.umath_tests import inner1d
from Utils.Utils import cosine_distance, softmax


class LangModule(object):
    """
    Language module for scene grapher
    """

    def __init__(self):
        """
        Initialize language module of scene grapher
        """
        # get singleton instance of word embed object
        self.word_embed_obj = WordEmbd()

    def predict(self, word1, word2, w, b):
        """
        Given subject - worda and object - wordb, returns array of predicate likelihood (each entry is predicate-id)
        :param word1: string - subject word
        :param word2:  string - object word
        :param w, b: external parameters
        :return: numpy array - array of predicate likelihood
        """
        # get W and B

        # Get embedded vectors
        embed_word1 = self.word_embed_obj.word2vec(word1)
        embed_word2 = self.word_embed_obj.word2vec(word2)
        embed = np.concatenate((embed_word1, embed_word2), axis=1)

        # calc f
        f = np.dot(w, embed.T) + b.reshape(-1, 1)
        # return softmax(f.T)
        return f.T

    def word_embed(self, words):
        """
        convert to word embedding
        :param words: single word or array of words
        :return: single word embedding or array of word embedding
        """
        return self.word_embed_obj.word2vec(words)

    def relation_embed(self, word_a_embed, word_b_embed):
        """
        convert word embedding relation embedding
        :param word_a_embed: embedding of subjects in the relationship
        :param word_b_embed: embedding of objects in the relationship
        :return: embedding of a relationship
        """
        return np.concatenate((word_a_embed, word_b_embed), axis=1)

    def distance(self, r1_a_embed, r1_b_embed, r1_predicate_embed, r2_a_embed, r2_b_embed, r2_predicate_embed):
        """
        Return an array of cosine distance between r1[i] and r2[i]
        :param r1_a_embed:  embedding of r1 subjects
        :param r1_b_embed: embedding of r1 objects
        :param r1_predicate_embed: embedding of r1 predicates
        :param r2_a_embed: embedding of r2 subjects
        :param r2_b_embed: embedding of r2 objects
        :param r2_predicate_embed: embedding of r2 predicates
        :return: cosine distance
        """
        dist = cosine_distance(r1_a_embed, r2_a_embed)
        dist += cosine_distance(r1_b_embed, r2_b_embed)
        dist += cosine_distance(r1_predicate_embed, r2_predicate_embed)
        return dist

    def likelihood(self, r_embed, w, b, predicate_ids):
        """
        Get the likelihood of a relationship
        :param r_embed: embedding of the relationship
        :param w: language module params
        :param b: language module params
        :param predicate_ids: ids of relationship predicates
        :return: array of likelihood
        """
        return inner1d(w[predicate_ids], r_embed).reshape(-1, 1) + b[predicate_ids].reshape(-1, 1)

    def cost_and_gradient(self, x, R1, R2, coeff_l=0.05, coeff_k=0.002):
        """
        Calculate the cost and the gradient with respect to language model parameters

        Loss(R) = K(R) + L(R)
                                                                                            |f(Ri) - f(Rj)|
        K(R) is the variance of projection function - K(R) = variance(Dij) =  variance({------------------------})
                                                                                         cosine_distance(Ri, Rj)

        L(R) is the likelihood of relationship - L(R) = Sum-ij(max{f(Ri) - f(Rj) + 1, 0})
                                                    Rj occurs more frequently than Ri
        :param coeff_k:
        :param coeff_l:
        :param x: numpy matrix - language module parameters
        :param R1: Data Object represents group of relationships to compare to R2 group
        :param R2: Data Object represents group of relationships to compare to R1 group
        :return: cost and gradient for each parameter in x
        """
        # Convert X to actual module parameters
        W = x[:, 0:-1]
        B = x[:, -1]

        ### Loss and gradient of K

        # calc R1 embedded words (concat(word2vec(worda), word2vec(wordb))
        r1_embed_word1 = self.word_embed.word2vec(R1.worda)
        r1_embed_word2 = self.word_embed.word2vec(R1.wordb)
        r1_embed = np.concatenate((r1_embed_word1, r1_embed_word2), axis=1)
        # calc f of R1
        r1_f = inner1d(W[R1.predicate_ids], r1_embed).reshape(-1, 1) + B[R1.predicate_ids].reshape(-1, 1)

        # calc R2 embedded words (concat(word2vec(worda), word2vec(wordb))
        r2_embed_word1 = self.word_embed.word2vec(R2.worda)
        r2_embed_word2 = self.word_embed.word2vec(R2.wordb)
        r2_embed = np.concatenate((r2_embed_word1, r2_embed_word2), axis=1)
        # calc f of R2
        r2_f = inner1d(W[R2.predicate_ids], r2_embed).reshape(-1, 1) + B[R2.predicate_ids].reshape(-1, 1)

        # cosine distance between R1 and R2 (sum of cosine distance between the words and predicates)

        dist = cosine_distance(r1_embed_word1, r2_embed_word1)
        dist += cosine_distance(r1_embed_word2, r2_embed_word2)
        dist += cosine_distance(self.word_embed.word2vec(R1.predicate), self.word_embed.word2vec(R2.predicate))

        # calc D
        #               |f(Ri) - f(Rj)|^2
        #       D = ------------------------
        #             cosine_distance(Ri, Rj)
        #
        f_sub = np.subtract(r2_f, r1_f)
        D = np.divide(f_sub, dist.reshape(-1, 1))
        D = np.multiply(D, f_sub)
        avgD = np.average(D)
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

        grad_w_coeff = 4 * np.divide(np.subtract(D, avgD), dist.reshape(-1, 1)) / D.shape[0]
        grad_w_coeff = np.multiply(grad_w_coeff, f_sub)
        grad_w_k = np.zeros(W.shape)
        np.add.at(grad_w_k, R1.predicate_ids, - grad_w_coeff * r1_embed)
        np.add.at(grad_w_k, R2.predicate_ids, grad_w_coeff * r2_embed)

        # calc K gradient B - nothing to do (always zero)
        grad_b_k = np.zeros(B.shape).reshape(-1, 1)
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
        grad_w_l = np.zeros(W.shape)
        l_r1_coeffs = np.copy(l_loss)
        l_r1_coeffs[l_r1_coeffs != 0] = 1.0 / l_loss.shape[0]
        l_r1_coeffs = np.multiply(l_r1_coeffs, -1 * l_coeff)
        np.add.at(grad_w_l, R1.predicate_ids, l_r1_coeffs * r1_embed)
        l_r2_coeffs = np.copy(l_loss)
        l_r2_coeffs[l_r2_coeffs != 0] = 1.0 / l_loss.shape[0]
        l_r2_coeffs = np.multiply(l_r2_coeffs, l_coeff)
        np.add.at(grad_w_l, R2.predicate_ids, l_r2_coeffs * r2_embed)

        # L grad b
        grad_b_l = np.zeros(B.shape).reshape(-1, 1)
        np.add.at(grad_b_l, R1.predicate_ids, l_r1_coeffs)
        np.add.at(grad_b_l, R2.predicate_ids, l_r2_coeffs)

        ### Calc C
        # W grad C
        C = 0
        # predicate_features, subject_probabilities, object_probabilities  = visual.extract_features(R1)

        ### total loss and grad
        loss = coeff_k * K + coeff_l * L
        grad_w = coeff_k * grad_w_k + coeff_l * grad_w_l
        grad_b = coeff_k * grad_b_k + coeff_l * grad_b_l
        grad = np.concatenate((grad_w, grad_b.reshape(-1, 1)), axis=1)

        return loss, grad


if __name__ == "__main__":
    embed = LangModule(70, 50)
    embed.predict("the", "hello")
    print("Debug")
