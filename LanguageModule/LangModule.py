import numpy as np
import cPickle
from WordEmbd import  WordEmbd
from scipy import spatial

class LangModule(object):

    def __init__(self, nof_predicats, embed_vector_dim):

        self.nof_predicates = nof_predicats
        self.embed_vector_dim = embed_vector_dim
        ## Module Parameters
        # W (DIM: predicats X 2embed_vector_dim)
        self.W = np.zeros((self.nof_predicates, 2 * self.embed_vector_dim))
        # B (DIM: predicates)
        self.B = np.zeros(self.nof_predicates)

        # get singelton instance of word embed object
        self.word_embed = WordEmbd()

    def predict(self, word1, word2):
        # Get embedded vectors
        embed_word1 = self.word_embed.word2vec(word1)
        embed_word2 = self.word_embed.word2vec(word2)
        embed = np.concatenate((embed_word1, embed_word2), axis=0)

        # calc f
        f = np.dot(self.W, embed) + self.B
        return f

    def get_weights(self):
        return np.concatenate((self.W, self.B.reshape(-1, 1)), axis=1)


    def cost_and_gradient(self, x, R1, R2):
        W = x[:, 0:-1]
        B = x[:, -1]

        ### K loss and gradient
        # calc R1 embed
        r1_embed_word1 = self.word_embed.word2vec(R1.worda)
        r1_embed_word2 = self.word_embed.word2vec(R1.wordb)
        r1_embed = np.concatenate((r1_embed_word1, r1_embed_word2), axis=0)
        # calc R1 f
        f = np.dot(W, r1_embed) + B
        r1_f = f[R1.predicate]

        # calc R2 embed
        r2_embed_word1 = self.word_embed.word2vec(R2.worda)
        r2_embed_word2 = self.word_embed.word2vec(R2.wordb)
        r2_embed = np.concatenate((r2_embed_word1, r2_embed_word2), axis=0)
        # calc R2 f
        f = np.dot(W, r2_embed) + B
        r2_f = f[R2.predicate]

        # cosine distance between R1 and R2
        cosine_distance = spatial.distance.cosine(r1_embed_word1, r2_embed_word1)
        cosine_distance += spatial.distance.cosine(r1_embed_word2, r2_embed_word2)
        # FIXME cosine_distance += spatial.distance.cosine(R1.predicate, R2.predicate)

        # calc D
        f_sub = np.subtract(r2_f, r1_f)
        D = np.divide(np.power(f_sub, 2), cosine_distance)

        # calc loss - variance of D
        loss = np.var(D)

        # calc gradient W
        grad_w_coeff = 4 * np.divide(np.subtract(D, np.average(D)), cosine_distance)
        grad_w_coeff = np.multiply(grad_w_coeff, np.subtract(r1_f, r2_f))
        grad_w = np.zeros(W.shape)
        grad_w[R1.predicate] += grad_w_coeff * r1_embed
        grad_w[R2.predicate] -= grad_w_coeff * r2_embed

        # calc gradient B
        grad_b = np.zeros(B.shape)


        ### L loss and gradient
        l_coeff = np.ones(R1.instances.shape)
        l_coeff[R1.instances < R2.instances] = -1
        l_coeff[R1.instances == R2.instances] = 0
        l_loss = np.maximum(np.multiply(l_coeff, f_sub) + 1, 0)
        loss += np.sum(l_loss)

        # grad W
        l_r1_coeffs = np.copy(l_loss)
        l_r1_coeffs[l_r1_coeffs != 0] = 1
        l_r1_coeffs = np.multiply(l_r1_coeffs, -1 * l_coeff)
        grad_w[R1.predicate] *= l_r1_coeffs * r1_embed
        l_r2_coeffs = np.copy(l_loss)
        l_r2_coeffs[l_r2_coeffs != 0] = 1
        l_r2_coeffs = np.multiply(l_r2_coeffs, l_coeff)
        grad_w[R2.predicate] += l_r2_coeffs * r2_embed

        # grad b - nothing to do

        ### total grad
        grad = np.concatenate((grad_w, grad_b.reshape(-1, 1)), axis=1)

        return loss, grad

if __name__ == "__main__":
    embed = LangModule(70, 50)
    embed.predict("the", "hello")
    print("Debug")