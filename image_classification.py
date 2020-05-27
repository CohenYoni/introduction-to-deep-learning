"""
Developers:
Yoni Cohen
May Hagbi
"""

import tensorflow as tf
import numpy as np
import argparse
import logging
import pickle
import sys
import cv2
import os

CIFAR100_FOLDER_PATH = 'cifar_100_python'
CIFAR100_META_FILE_PATH = os.path.join(CIFAR100_FOLDER_PATH, 'meta')
WORD_VECTORS_FILE_PATH = 'wiki-news-300d-1M.vec'
DEFAULT_MODEL_PATH = 'image_model.h5'

logging.basicConfig(stream=sys.stdout, format='%(asctime)s | %(message)s', level=logging.INFO)


def get_user_cli_args():
    """
    Get I/O arguments from user passed from command line.
    :return: user arguments parsed by ArgumentParser
    """
    parser = argparse.ArgumentParser(description='Find the sentence most similar to given query')
    parser.add_argument('--task', choices=TASKS.keys(), help='/'.join(TASKS.keys()), required=True)
    parser.add_argument('--model', help='trained model file global path', default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def get_train_and_test_data():
    """
    get cifar100 dataset (data and labels)
    :return: train and test samples divided to 80/20
    """
    train, test = tf.keras.datasets.cifar100.load_data(label_mode="fine")
    return train, test


def get_index_to_word_label_mapper(dictionary_file_path):
    """
    Get index-to-word mapper of cifar100 dataset
    :param dictionary_file_path: cifar100 meta binary file
    :return: list of words, index i maps to word label #i
    """
    with open(dictionary_file_path, 'rb') as fp:
        label_dictionary = pickle.load(fp, encoding='utf-8')
    return label_dictionary['fine_label_names']


def map_label_index_to_word(labels):
    """
    map every label index to word
    :param labels: numpy array of label indexes
    :return: numpy array of label words
    """
    label_dictionary = get_index_to_word_label_mapper(CIFAR100_META_FILE_PATH)
    return np.array([label_dictionary[x[0]] for x in labels])


def load_fast_text_vectors(vectors_file_path):
    """
    Load word embedding vectors from a file
    :param vectors_file_path: an absolute path of the file
    :return: numpy array of vectors
    """
    with open(vectors_file_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fp:
        length, dimension = map(int, fp.readline().split())  # read header
        logging.info(f'{vectors_file_path}: length={length}, dimension={dimension}')
        data = {}
        for line in fp:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return data


def pixel_treatment(data):
    """
    normalize the data
    :param data: numpuy array of data
    :return: normalized data
    """
    return np.array(data, dtype="float") / 255.0


def map_words_labels_to_word_vector(words_labels):
    """
    map every label to word vector
    :param words_labels: numpy array of word(s) labels
    :return: numpy array of word vectors
    """
    logging.info('loading vectors...')
    word_vectors = load_fast_text_vectors(WORD_VECTORS_FILE_PATH)
    # TODO: if labels contains '_' we need to split it and calculate the avg of wv
    return []


def pre_processing(dataset):
    """
    performs pre processing of data and labels
    :param dataset: data and labels
    :return: pre processed data and labels
    """
    data, labels = dataset
    data = pixel_treatment(data)
    labels = map_label_index_to_word(labels)
    labels = map_words_labels_to_word_vector(labels)
    return data, labels


def training(data, labels, model_file_path):
    pass


def train_task_handler(user_args):
    logging.info('loading dataset...')
    train, test = get_train_and_test_data()
    logging.info('performing train pre processing...')
    train_data, train_labels = pre_processing(train)
    logging.info('performing test pre processing...')
    test_data, test_labels = pre_processing(test)
    logging.info('start training..')
    training(train_data, train_labels, user_args.model)
    logging.info('train task done!')


def read_image(image_path):
    """
    read image and convert it to 32*32*3 pixels
    :param image_path: an image path
    :return: numpy array of 32*32*3 pixels image
    """
    image = cv2.imread(image_path)
    return cv2.resize(image, (32, 32))  # to get images sized 3072


def test_task_handler(user_args):
    # TODO: print 3 text labels (words) most relevant to this image (from the most relevant to the least relevant):
    #   <label 1> <label 2> <label 3>
    pass


# mapping task to function
TASKS = {
    'train': train_task_handler,
    'test': test_task_handler
}


def main():
    """
    Entry point of the script
    """
    args = get_user_cli_args()
    task_handler = TASKS[args.task]  # get the task handler function from global dictionary mapper
    task_handler(args)


if __name__ == '__main__':
    main()
