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
    GLOVE_FILE_NAME = 'glove.6B.50d.txt'
    EMBED_PICKLE_FILE = 'glove50.p'
    VOCAB_PICKLE_FILE = 'glove50_vocab.p'
    WORD_INDEX_PICKLE_FILE = 'glove50_word_index.p'

    def __init__(self):
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
        self.loadWordEmbd()


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

    def loadWordEmbd(self):
        """
        Load / prepare Word embedding module
        :return:
        """
        if os.path.isfile(WordEmbd.EMBED_PICKLE_FILE) and os.path.isfile(WordEmbd.VOCAB_PICKLE_FILE) and os.path.isfile(WordEmbd.WORD_INDEX_PICKLE_FILE):
            #if already saved to pickle files, load it
            embed_file = file(WordEmbd.EMBED_PICKLE_FILE, "rb")
            self.embed = cPickle.load(embed_file)
            embed_file.close()

            vocab_file = file(WordEmbd.VOCAB_PICKLE_FILE, "rb")
            self.vocab = cPickle.load(vocab_file)
            vocab_file.close()

            word_index_file = file(WordEmbd.WORD_INDEX_PICKLE_FILE, "rb")
            self.word_index = cPickle.load(word_index_file)
            word_index_file.close()

            self.vocab_size = self.embed.shape[0]
            self.vector_dim = self.embed.shape[1]

        else:
            # if pickle files, does not exist - prepare module.
            # load data
            print "Load Data"
            vocab, embd, word_index = self.loadGlove(WordEmbd.GLOVE_FILE_NAME)
            self.vocab_size = len(vocab)
            self.vector_dim = len(embd[0])

            # load to class fields
            self.embed = np.asarray(embd)
            self.vocab = vocab
            self.word_index = word_index

            #Save picke files
            print "Save Embed Words"
            embed_file = open(WordEmbd.EMBED_PICKLE_FILE, "wb")
            cPickle.dump(self.embed, embed_file, 0)
            embed_file.close()

            print "Save Vocab"
            vocab_file = open(WordEmbd.VOCAB_PICKLE_FILE, "wb")
            cPickle.dump(self.vocab, vocab_file, 0)
            vocab_file.close()

            print "Save Word Index"
            word_index_file = open(WordEmbd.WORD_INDEX_PICKLE_FILE, "wb")
            cPickle.dump(self.word_index, word_index_file, 0)
            word_index_file.close()

    def word2vec(self, word):
        """
        Convet words to embedded vector representation. Supprt both single word and list of words
        :param word: word or list of words
        :return: embedded vector or embedded matrix
        """
        if isinstance(word, np.ndarray):
            indices = []
            for elem in word:
                if self.word_index.has_key(elem):
                    indices.append(self.word_index[elem])
                else:
                    indices.append(self.word_index["unknown"])
        else:
            indices =  self.word_index[word]
        return self.embed[indices].astype("float32")

    def embed_vec_dim(self):
        return self.vector_dim

    def cosine_distance(self, embed_word_a, embed_word_b):
        """
        Calc the cosine distance between to embedded words
        :param embed_word_a:
        :param embed_word_b:
        :return:
        """
        raise NameError("TBD")

if __name__ == "__main__":
    embed = WordEmbd()
    print("Debug")
