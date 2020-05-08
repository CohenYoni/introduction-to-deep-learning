"""
Developers:
Yoni Cohen
May Hagbi
"""

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import argparse
import logging
import string
import re

MOST_SIMILAR_FILE_NAME = 'most_similar.txt'
WORD_VECTORS_FILE_PATH = 'wiki-news-300d-1M.vec'

logging.basicConfig(format='%(asctime)s | %(message)s', level=logging.INFO)


def is_alphabetic(word_str):
    """
    Return true if the word contains only alphabetic characters
    :param word_str: a string representing a word
    """
    return re.match(r'^[a-zA-Z]+$', word_str) is not None


def is_alphanumeric(word_str):
    """
    Return true if the word contains only alphanumeric characters
    :param word_str: a string representing a word
    """
    return re.match(r'^[a-zA-Z0-9]*[a-zA-Z][a-zA-Z0-9]*$', word_str) is not None


def remove_punctuation(word_str):
    """
    Remove punctuation from a string
    """
    return re.sub(rf'[{string.punctuation}]', '', word_str)


def get_user_cli_args():
    """
    Get I/O arguments from user passed from command line.
    :return: user arguments parsed by ArgumentParser
    """
    parser = argparse.ArgumentParser(description='Find the sentence most similar to given query')
    parser.add_argument('--query', help='query file global path')
    parser.add_argument('--text', help='text file global path')
    parser.add_argument('--task', choices=TASKS.keys(), help='/'.join(TASKS.keys()), required=True)
    parser.add_argument('--data', help='training dataset in .csv format')
    parser.add_argument('--model', help='trained model file global path')
    return parser.parse_args()


def read_dataset(dataset_file_path):
    """
    Read dataset from a csv file
    :param dataset_file_path: an absolute path for dataset file
    :return: a dictionary of numpy arrays contains tags (query) and description (sentence)
    """
    dataset = pd.read_csv(dataset_file_path)
    return {
        'description': np.asarray(dataset['description'][:5]),  # TODO: remove slicing
        'tags': np.asarray(dataset['tags'][:5])  # TODO: remove slicing
    }


def train_task_handler(user_args):
    """
    Run training task
    :param user_args: user cli arguments
    """
    assert user_args.data, '--data argument is missing'
    assert user_args.model, '--model argument is missing'
    logging.info('reading dataset...')
    dataset = read_dataset(user_args.data)
    logging.info('pre processing sentences...')
    processed_sentences = np.asarray(list(map(sentence_pre_processing, dataset['description'])))
    logging.info('pre processing tags...')
    processed_tags = np.asarray(list(map(sentence_pre_processing, dataset['tags'])))
    logging.info('loading vectors...')
    word_vectors = load_vectors(WORD_VECTORS_FILE_PATH)
    logging.info('converting sentences to average vectors...')
    sentences_avg_vectors = get_average_vectors(processed_sentences, word_vectors)
    logging.info('converting tags to average vectors...')
    tags_avg_vectors = get_average_vectors(processed_tags, word_vectors)
    training(sentences_avg_vectors, tags_avg_vectors, user_args.model)


def training(data, labels, model_file_path):
    """
    Training the model
    Stages: model definition, model compilation, model fitting (training), model evaluation
    :param data: numpy array of sentences average vectors
    :param labels: numpy array of tags average vectors
    :param model_file_path: an absolute path for file who saves the model structure and weights
    """
    # TODO: dataset splitting (70/30)
    # TODO: model definition (300 dimension input and output layers)
    # TODO: model compilation
    # TODO: model fitting
    # TODO: model evaluation
    # TODO: save model in file
    pass


def get_average_vectors(array_of_words_array, word_vectors):
    """
    Get average vector for each sentence by converting each word to FastText vector and calculate average
    :param array_of_words_array: numpy array of sentences, each sentence is a numpy array of words
    :param word_vectors: FastText word embedding vector
    :return: numpy array of average vector (vector length is 300)
    """
    sentences_avg_vectors = []
    for sentence in array_of_words_array:
        # get sentence vector and remove None (None element is an unknown word, it's not in word vector)
        sentence_vectors = map(lambda word: word_vectors.get(word, None), sentence)
        vectors_without_none = np.asarray(list(filter(lambda vector: vector is not None, sentence_vectors)))
        # calculate vector average
        sentence_avg = np.average(vectors_without_none, axis=0)
        sentences_avg_vectors.append(sentence_avg)
    return np.asarray(sentences_avg_vectors)


def load_vectors(vectors_file_path):
    """
    Load word embedding vectors from a file
    :param vectors_file_path: an absolute path of the file
    :return: numpy array of vectors
    """
    with open(vectors_file_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fp:
        length, dimension = map(int, fp.readline().split())  # read header
        data = {}
        for line in fp:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return data


def test_task_handler(user_args):
    """
    Run testing task
    :param user_args: user cli arguments
    """
    assert user_args.query, '--query argument is missing'
    assert user_args.text, '--text argument is missing'


def sentence_pre_processing(raw_sentence):
    """
    Pre processing step on a sentence includes lower case, remove punctuation, remove stop words and filter alphanumeric
    :param raw_sentence: a sentence from raw data
    :return: numpy array of processed sentence
    """
    words = np.asarray(word_tokenize(raw_sentence.lower()))  # lower case and tokenization
    punctuation_removed = map(remove_punctuation, words)  # remove punctuation
    stopwords_filtered = filter(lambda word: word not in ALL_STOPWORDS, punctuation_removed)  # stop word removal
    return np.asarray(list(filter(is_alphanumeric, stopwords_filtered)))


# mapping task to function
TASKS = {
    'train': train_task_handler,
    'test': test_task_handler
}

ALL_STOPWORDS = stopwords.words()


def main():
    """
    Entry point of the script
    """
    args = get_user_cli_args()
    task_handler = TASKS[args.task]  # get the task handler function from global dictionary mapper
    task_handler(args)


if __name__ == '__main__':
    main()
