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

    def __init__(self, nof_predicats, embed_vector_dim):
        """
        Initilize language module of scene grapher
        :param nof_predicats: nof predicates that should be represented by this module
        :param embed_vector_dim: dimensions of the embedded word
        """

        self.nof_predicates = nof_predicats
        self.embed_vector_dim = embed_vector_dim
        ## Module Parameters
        # W (DIM: predicats X 2embed_vector_dim)
        # self.W = np.zeros((self.nof_predicates, 2 * self.embed_vector_dim))
        self.W = np.random.rand(self.nof_predicates, 2 * self.embed_vector_dim)
        # B (DIM: predicates)
        self.B = np.random.rand(self.nof_predicates)

        # get singelton instance of word embed object
        self.word_embed = WordEmbd()

    def predict(self, word1, word2, x=None):
        """
        Given subject - worda and object - wordb, returns array of predicate likelihood (each entry is predicate-id)
        :param word1: string - subject word
        :param word2:  string - object word
        :param x: external parameters
        :return: numpy array - array of predicate likelihood
        """
        # get W and B
        if x is not None:
            W = x[:, 0:-1]
            B = x[:, -1]
        else:
            W = self.W
            B = self.B

        # Get embedded vectors
        embed_word1 = self.word_embed.word2vec(word1)
        embed_word2 = self.word_embed.word2vec(word2)
        embed = np.concatenate((embed_word1, embed_word2), axis=1)

        # calc f
        f = np.dot(W, embed.T) + B.reshape(-1, 1)
        # return softmax(f.T)
        return f.T

    def get_weights(self):
        """
        return module parameters,
        :return: numpy matrix - module paramters,
        """
        return np.concatenate((self.W, self.B.reshape(-1, 1)), axis=1)

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
        #C = 0
        #for R in R1:
        #    C +=  np.max(max_v(R, R2) - v(R) + 1, 0)

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
