import os
import os.path
import numpy as np
import sys
sys.path.append("..")
from DesignPatterns.Singleton import Singleton
import cPickle


class WordEmbd(object):
    """
    Word to vector embeddin, using GLOVE
    """
    __metaclass__ = Singleton

    # fields
    # GLOVE_FILE_NAME = 'LanguageModule/glove.6B.300d.txt'
    GLOVE_FILE_NAME_50 = 'glove.6B.50d.txt'
    GLOVE_FILE_NAME_300 = 'glove.6B.300d.txt'
    EMBED_PICKLE_FILE_50 = 'glove50.p'
    EMBED_PICKLE_FILE_300 = 'glove300.p'

    def __init__(self, word_embed_size):
        ## init fields

        # number of words in vocabulary
        self.vocab_size = None
        # word embedded dimensions
        self.vector_dim = None
        # matrix of vocab_size X vector_dim
        self.embed = None
        # word to word_index
        self.word_index = None

        ## load Glove
        self.loadWordEmbd(word_embed_size)


    def loadGlove(self, filename):
        """
        Load Glove Word Embedding (expecting pre-traind data to be stored in filename
        :param filename: that path to the pre-trained data
        :return: vacobulary and embeddings
        """
        vocab = []
        word_index = {}
        embd = []
        file = open(filename, 'rb')
        index = 0
        for line in file.readlines():
            row = line.strip().split(' ')
            vocab.append(row[0])
            embd.append(row[1:])
            word_index[row[0]] = index
            index += 1
        print('Loaded GloVe!')
        file.close()
        return vocab, embd, word_index

    def loadWordEmbd(self, word_embed_size):
        """
        Load / prepare Word embedding module
        :return:
        """
        if word_embed_size == 50:
            glove_file_name = WordEmbd.GLOVE_FILE_NAME_50
            embed_pickle_file = WordEmbd.EMBED_PICKLE_FILE_50
        else:
            glove_file_name = WordEmbd.GLOVE_FILE_NAME_300
            embed_pickle_file = WordEmbd.EMBED_PICKLE_FILE_300

        if os.path.isfile(embed_pickle_file):
            #if already saved to pickle files, load it
            embed_file = file(embed_pickle_file, "rb")
            self.embed = cPickle.load(embed_file)
            self.vocab = cPickle.load(embed_file)
            self.word_index = cPickle.load(embed_file)
            embed_file.close()

            self.vocab_size = self.embed.shape[0]
            self.vector_dim = self.embed.shape[1]

        else:
            # if pickle files, does not exist - prepare module.
            # load data
            print "Load Data"
            vocab, embd, word_index = self.loadGlove(glove_file_name)
            self.vocab_size = len(vocab)
            self.vector_dim = len(embd[0])

            # load to class fields
            self.embed = np.asarray(embd)
            self.vocab = vocab
            self.word_index = word_index

            #Save picke files
            print "Save Embed Words"
            embed_file = open(embed_pickle_file, "wb")
            cPickle.dump(self.embed, embed_file, 0)
            cPickle.dump(self.vocab, embed_file, 0)
            cPickle.dump(self.word_index, embed_file, 0)
            embed_file.close()


    def word2vec(self, word):
        """
        Convet words to embedded vector representation. Supprt both single word and list of words
        :param word: word or list of words
        :return: embedded vector or embedded matrix
        """
        if isinstance(word, np.ndarray):
            index = 0
            words_vec = np.zeros((len(word), self.vector_dim))
            for elem in word:
                for single in elem.split():
                    # temp work-around for phrase
                    if self.word_index.has_key(single):
                        words_vec[index] += self.embed[self.word_index[single]].astype("float64")
                    else:
                        words_vec[index] += self.embed[self.word_index["unknown"]].astype("float64")
                index += 1
        else:
            indices =  self.word_index[word]
            words_vec = self.embed[indices].astype("float32")
        return words_vec

    def embed_vec_dim(self):
        return self.vector_dim

if __name__ == "__main__":
    embed = WordEmbd()
    print("Debug")
