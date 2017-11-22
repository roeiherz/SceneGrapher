from FeaturesExtraction.Utils.data import word2vec_mapping_func
import sys
sys.path.append("..")
from Data.VisualGenome.local import GetAllRegionDescriptions
import os
from gensim.models import Word2Vec
from Utils.Logger import Logger
from FilesManager.FilesManager import FilesManager
import cPickle
import re

__author__ = 'roeih'


def multi_word_replace(text, wordDic):
    """
    take a text and replace words that match a key in a dictionary with
    the associated value, return the changed text
    """
    rc = re.compile('|'.join(map(re.escape, wordDic)))
    def translate(match):
        return wordDic[match.group(0)]
    return rc.sub(translate, text)


def get_corpus_from_image_captioning(debug=1):
    """
    This function will get the whole image captioning from the images and saves them as a corpus
    :return: train_sentences and test_sentences lists
    """

    # Get region_interest depends on debug flag
    if debug:
        region_interest = cPickle.load(open("../region_interst10.p"))
    else:
        # load entities
        region_interest = GetAllRegionDescriptions("/specific/netapp5_2/gamir/DER-Roei/SceneGrapher/Data/VisualGenome/data")

    # idx_to_ids = FilesManager().load_file("data.visual_genome.idx_to_ids")
    img_id_to_split = FilesManager().load_file("data.visual_genome.img_id_to_split")

    # Check if pickles are already created
    train_corpus_path = FilesManager().get_file_path("language_module.word2vec.train_corpus")
    test_corpus_path = FilesManager().get_file_path("language_module.word2vec.test_corpus")

    if os.path.isfile(train_corpus_path) and os.path.isfile(test_corpus_path):
        Logger().log('Files {0},{1} are already exist'.format(train_corpus_path, test_corpus_path))
        train_corpus = FilesManager().load_file("language_module.word2vec.train_corpus")
        test_corpus = FilesManager().load_file("language_module.word2vec.test_corpus")
        return train_corpus, test_corpus

    # index for saving
    ind = 1
    Logger().log("Start creating corpus for train and test from image captioning")
    word2vec_mapping = word2vec_mapping_func()
    train_sentences = []
    test_sentences = []
    for region in region_interest:
        for phrase_data in region:
            try:

                # Get the phrase
                phrase_txt = phrase_data.phrase.replace(".", "").replace("\n", "").replace(",", "").rstrip().lstrip()
                phrase_txt_replaced = multi_word_replace(phrase_txt, word2vec_mapping)
                sentence_lst = phrase_txt_replaced.split(" ")
                img_id = phrase_data.image.id
                Logger().log("Image id: {0}".format(img_id))
                # Write train corpus
                if img_id_to_split[img_id] == 0:
                    train_sentences.append(sentence_lst)

                # Write test corpus
                if img_id_to_split[img_id] == 2:
                    test_sentences.append(sentence_lst)

                ind += 1
            except Exception as e:
                print("Problem with {0} in index: {1} image id: {2}".format(e, ind, img_id))

    Logger().log("Finish to create train and test corpus")

    # Save pickle files
    FilesManager().save_file("language_module.word2vec.train_corpus", train_sentences)
    FilesManager().save_file("language_module.word2vec.test_corpus", test_sentences)
    return train_sentences, test_sentences


def get_train_and_test_word2vec_models(train_sentences, test_sentences):
    """
    This function save or load the training and testing word2vec models
    :return:
    """

    # Check if pickles are already created
    train_model_path = FilesManager().get_file_path("language_module.word2vec.train_model")
    test_model_path = FilesManager().get_file_path("language_module.word2vec.test_model")

    if os.path.isfile(train_model_path) and os.path.isfile(test_model_path):
        Logger().log('Files {0},{1} are already exist'.format(train_model_path, test_model_path))
        train_model = FilesManager().load_file("language_module.word2vec.train_model")
        test_model = FilesManager().load_file("language_module.word2vec.test_model")
        return train_model, test_model

    # Train Model
    train_model = Word2Vec(train_sentences, min_count=1, size=10)
    # Save Model
    train_model.save(FilesManager().get_file_path("language_module.word2vec.train_model"))

    # Test Model
    test_model = Word2Vec(test_sentences, min_count=1, size=10)
    # Save Model
    test_model.save(FilesManager().get_file_path("language_module.word2vec.test_model"))

    return train_model, test_model

if __name__ == '__main__':

    # Get data
    train_sentences, test_sentences = get_corpus_from_image_captioning()
    # Get models
    train_model, test_model = get_train_and_test_word2vec_models(train_sentences=train_sentences,
                                                                 test_sentences=test_sentences)

    exit()
    # Summarize the loaded model
    print(train_model)
    # Summarize vocabulary
    words = list(train_model.wv.vocab)
    print(words)
    # Access vector for one word
    print(train_model['sentence'])
    # Save model
    train_model.save('model.bin')
    # load model
    new_model = Word2Vec.load('model.bin')
    print(new_model)
