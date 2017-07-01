import os
import os.path
import numpy as np
import sys

from FilesManager.FilesManager import FilesManager

sys.path.append("..")
from DesignPatterns.Singleton import Singleton


class WordEmbd(object):
    """
    Word to vector embeddin, using GLOVE
    """
    __metaclass__ = Singleton

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

        # file-manager
        self.filemanager = FilesManager()

        ## load Glove
        self.loadWordEmbd(word_embed_size)


    def loadGlove(self, filename, word_embed_size):
        """
        Load Glove Word Embedding (expecting pre-traind data to be stored in filename
        :param filename: that path to the pre-trained data
        :return: vacobulary and embeddings
        """
        vocab = []
        word_index = {}
        embd = []
        lines = self.filemanager.load_file("word_embedding.glove.original" + str(word_embed_size))
        index = 0
        for line in lines:
            row = line.strip().split(' ')
            vocab.append(row[0])
            embd.append(row[1:])
            word_index[row[0]] = index
            index += 1
        print('Loaded GloVe!')

        return vocab, embd, word_index

    def loadWordEmbd(self, word_embed_size):
        """
        Load / prepare Word embedding module
        :return:
        """
        if self.filemanager.file_exist("word_embedding.glove.embed" + str(word_embed_size)):
            data = self.filemanager.load_file("word_embedding.glove.embed" + str(word_embed_size))

            #if already saved to pickle files, load it
            self.embed = data[0]
            self.vocab = data[1]
            self.word_index = data[2]

            self.vocab_size = self.embed.shape[0]
            self.vector_dim = self.embed.shape[1]

        else:
            # if pickle files, does not exist - prepare module.
            # load data
            print "Load Data"
            vocab, embd, word_index = self.loadGlove(word_embed_size)
            self.vocab_size = len(vocab)
            self.vector_dim = len(embd[0])

            # load to class fields
            self.embed = np.asarray(embd)
            self.vocab = vocab
            self.word_index = word_index

            #Save picke files
            print "Save Embed Words"
            data = [self.embed, self.vocab, self.word_index]
            self.filemanager.save_file("word_embedding.glove.embed" + str(word_embed_size))


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
    embed = WordEmbd(50)
    print("Debug")
