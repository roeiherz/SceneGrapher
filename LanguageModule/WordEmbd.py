import os
import os.path
import numpy as np
from DesignPatterns.Singleton import Singleton
import cPickle


class WordEmbd(object):
    __metaclass__ = Singleton

    # fields
    # GLOVE_FILE_NAME = 'LanguageModule/glove.6B.300d.txt'
    GLOVE_FILE_NAME = 'glove.6B.50d.txt'
    EMBED_PICKLE_FILE = 'glove50.p'
    VOCAB_PICKLE_FILE = 'glove50_vocab.p'

    def __init__(self):
        ## init fields

        # number of words in vocabulary
        self.vocab_size = None
        # word embedded dimensions
        self.vector_dim = None
        # matrix of vocab_size X vector_dim
        self.embed = None

        ## load Glove
        self.loadWordEmbd()


    def loadGlove(self, filename):
        """
        Load Glove Word Embedding (expecting pre-traind data to be stored in filename
        :param filename: that path to the pre-trained data
        :return: vacobulary and embeddings
        """
        vocab = []
        embd = []
        file = open(filename, 'rb')
        for line in file.readlines():
            row = line.strip().split(' ')
            vocab.append(row[0])
            embd.append(row[1:])
        print('Loaded GloVe!')
        file.close()
        return vocab, embd

    def loadWordEmbd(self):

        if os.path.isfile(WordEmbd.EMBED_PICKLE_FILE) and os.path.isfile(WordEmbd.VOCAB_PICKLE_FILE):
            embed_file = file(WordEmbd.EMBED_PICKLE_FILE, "rb")
            self.embed = cPickle.load(embed_file)
            embed_file.close()

            vocab_file = file(WordEmbd.VOCAB_PICKLE_FILE, "rb")
            self.vocab = cPickle.load(vocab_file)
            vocab_file.close()

            self.vocab_size = self.embed.shape[0]
            self.vector_dim = self.embed.shape[1]

        else:
            # load data
            vocab, embd = self.loadGlove(WordEmbd.GLOVE_FILE_NAME)
            self.vocab_size = len(vocab)
            self.vector_dim = len(embd[0])

            #convert to numpy matrix
            self.embed = np.asarray(embd)
            self.vocab = vocab

            #Save picke files
            embed_file = file(WordEmbd.EMBED_PICKLE_FILE, "wb")
            cPickle.dump(self.embed, embed_file, 0)
            embed_file.close()

            vocab_file = file(WordEmbd.VOCAB_PICKLE_FILE, "wb")
            cPickle.dump(self.vocab, vocab_file, 0)
            vocab_file.close()


if __name__ == "__main__":
    embed = WordEmbd()
    print("Debug")
