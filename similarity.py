"""
Developers:
Yoni Cohen
May Hagbi
"""

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from keras.layers.core import Dense
from keras.models import Sequential
from keras.models import load_model
from nltk.corpus import stopwords
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import logging
import string
import sys
import re
import os

WORD_VECTORS_FILE_PATH = 'wiki-news-300d-1M.vec'
MOST_SIMILAR_FILE_NAME = 'most_similar.txt'
GRAPH_PATH = 'training_history_graph.png'
TEST_PORTION_SIZE = 0.3

logging.basicConfig(stream=sys.stdout, format='%(asctime)s | %(message)s', level=logging.INFO)


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
    parser.add_argument('--model', help='trained model file global path', required=True)
    return parser.parse_args()


def read_dataset(dataset_file_path):
    """
    Read dataset from a csv file
    :param dataset_file_path: an absolute path for dataset file
    :return: a dictionary of numpy arrays contains tags (query) and description (sentence)
    """
    dataset = pd.read_csv(dataset_file_path)
    dataset.fillna('', inplace=True)
    return {
        'description': np.asarray(dataset['description']),
        'tags': np.asarray(dataset['tags'])
    }


def remove_sentences_without_tags(dataset):
    """
    Filter out missing data
    :param dataset: dataset from csv
    :return: correct dataset
    """
    filter_array = dataset['tags'] != ''
    dataset['description'] = dataset['description'][filter_array]
    dataset['tags'] = dataset['tags'][filter_array]
    return dataset


def train_task_handler(user_args):
    """
    Run training task
    :param user_args: user cli arguments
    """
    assert user_args.data, '--data argument is missing'
    logging.info('reading dataset...')
    dataset = read_dataset(user_args.data)
    logging.info('remove sentences without tags...')
    dataset = remove_sentences_without_tags(dataset)
    logging.info('sentences pre processing...')
    processed_sentences = np.asarray(list(map(sentence_pre_processing, dataset['description'])))
    logging.info('tags pre processing...')
    processed_tags = np.asarray(list(map(sentence_pre_processing, dataset['tags'])))
    logging.info('loading vectors...')
    word_vectors = load_vectors(WORD_VECTORS_FILE_PATH)
    logging.info('converting sentences to average vectors...')
    sentences_avg_vectors = get_average_vectors_from_array(processed_sentences, word_vectors)
    logging.info('converting tags to average vectors...')
    tags_avg_vectors = get_average_vectors_from_array(processed_tags, word_vectors)
    logging.info('start training..')
    training(data=tags_avg_vectors, labels=sentences_avg_vectors, model_file_path=user_args.model)
    logging.info('train task done!')


def training(data, labels, model_file_path):
    """
    Training the model
    Stages: model definition, model compilation, model fitting (training), model evaluation
    :param data: numpy array of sentences average vectors
    :param labels: numpy array of tags average vectors
    :param model_file_path: an absolute path for file who saves the model structure and weights
    """
    # train-test splitting
    logging.info('splitting...')
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels,
                                                                        test_size=TEST_PORTION_SIZE, shuffle=True,
                                                                        random_state=np.random.randint(1, high=1000))
    # model definition
    logging.info('defining the model...')
    model = Sequential()  # layers are stacked one upon each other
    model.add(Dense(300, activation="relu", input_dim=300))  # input layer, 300 is the size of an input vector
    # stacking takes care of matching output dimensions
    model.add(Dense(250, activation="relu"))
    model.add(Dense(200, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(200, activation="relu"))
    model.add(Dense(250, activation="relu"))
    model.add(Dense(300, activation=None))
    # model compilation
    logging.info('compiling the model...')
    model.compile(optimizer=SGD(lr=0.01), loss='mean_squared_error', metrics=['accuracy'])
    # model fitting
    logging.info('training...')
    training_history = model.fit(train_data, train_labels, epochs=50, batch_size=10)
    # model evaluation
    logging.info('evaluating...')
    test_loss, test_accuracy = model.evaluate(test_data, test_labels, batch_size=10)
    print(f'evaluating scores: loss = {test_loss}, accuracy = {test_accuracy}')
    logging.info('creating training graph...')
    create_training_graph(training_history, GRAPH_PATH,
                          test_history={'test_loss': test_loss, 'test_accuracy': test_accuracy})
    # save model to json and h5 files
    logging.info('save model...')
    save_model_to_json(model, model_file_path)
    save_model_to_h5(model, model_file_path)


def create_training_graph(training_history, image_path, test_history=None, with_validation=False):
    """
    Create loss and accuracy graphs
    :param training_history: a history object returned from model.fit function
    :param image_path: an absolute path for graph file
    :param test_history: a dictionary with test accuracy and test loss
    :param with_validation: mark that training_history contains validation data
    """
    # axises data
    epoch_axis = training_history.epoch
    accuracy_axis = training_history.history['accuracy']
    loss_axis = training_history.history['loss']
    # draw training data
    figure, (accuracy_line, loss_line) = plt.subplots(2)
    figure.suptitle('Training History')
    accuracy_line.set(xlabel='x - epochs', ylabel='y - accuracy')
    loss_line.set(xlabel='x - epochs', ylabel='y - loss')
    accuracy_line_label = 'train accuracy'
    loss_line_label = 'train loss'
    # draw test data
    if test_history:
        accuracy_line_label += f', test accuracy = {test_history["test_accuracy"]}'
        loss_line_label += f', test loss = {test_history["test_loss"]}'
    accuracy_line.plot(epoch_axis, accuracy_axis, label=accuracy_line_label)
    loss_line.plot(epoch_axis, loss_axis, label=loss_line_label)
    # draw validation data
    if with_validation:
        val_accuracy = training_history.history['val_accuracy']
        val_loss = training_history.history['val_loss']
        accuracy_line.plot(epoch_axis, val_accuracy, label='validation accuracy')
        loss_line.plot(epoch_axis, val_loss, label='validation loss')
    accuracy_line.legend()
    loss_line.legend()
    figure.savefig(image_path)
    plt.show()
    plt.close(figure)


def save_model_to_h5(model_obj, model_file_path):
    """
    Save complete model to h5 file
    :param model_obj: an object represents the model
    :param model_file_path: file path to save the model
    """
    if not model_file_path.endswith('.h5'):
        model_file_path = f'{model_file_path}.h5'
    model_obj.save(model_file_path)
    logging.info(f'Model saved to {model_file_path}')


def save_model_to_json(model_obj, model_file_path):
    """
    Save model structure to json file and weights to h5 file
    :param model_obj: an object represents the model
    :param model_file_path: file path to save the model
    """
    json_file_path = f'{model_file_path}.json'
    weights_file_path = f'{model_file_path}_json_weights.h5'
    # save model to json file
    model_json = model_obj.to_json()
    with open(json_file_path, 'w') as json_file:
        json_file.write(model_json)
    logging.info(f'Model saved to {json_file_path}')
    # save weights to h5 file
    model_obj.save_weights(weights_file_path)
    logging.info(f'Model weights saved to {weights_file_path}')


def get_average_vector(words_array, word_vectors):
    """
    Get average vector of sentence by converting it to FastText vector and calculate average
    :param words_array: a numpy array of words
    :param word_vectors: FastText word embedding vector
    :return: average vector (vector length is 300)
    """
    # get sentence vector and remove None (None element is an unknown word, it's not in word vector)
    sentence_vectors = map(lambda word: word_vectors.get(word, None), words_array)
    vectors_without_none = np.asarray(list(filter(lambda vector: vector is not None, sentence_vectors)))
    # calculate vector average
    sentence_avg = np.average(vectors_without_none, axis=0)
    return sentence_avg


def get_average_vectors_from_array(array_of_words_array, word_vectors):
    """
    Get average vector for each sentence by converting each word to FastText vector and calculate average
    :param array_of_words_array: a numpy array of sentences, each sentence is a numpy array of words
    :param word_vectors: FastText word embedding vector
    :return: numpy array of average vector (vector length is 300)
    """
    sentences_avg_vectors = []
    for sentence in array_of_words_array:
        # get average vector of one sentence
        sentence_avg = get_average_vector(sentence, word_vectors)
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
        logging.info(f'{vectors_file_path}: length={length}, dimension={dimension}')
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
    logging.info('loading model...')
    model = load_model(user_args.model)
    model.summary()
    logging.info(f'reading query from {user_args.query} file...')
    with open(user_args.query, 'r', errors='ignores', encoding='utf-8') as query_file:
        query = query_file.readline().replace('\n', '')
    logging.info(f'reading sentences from {user_args.text} file...')
    with open(user_args.text, 'r', errors='ignores', encoding='utf-8') as sentences_file:
        sentences = [sentence.replace('\n', '') for sentence in sentences_file.readlines()]
    sentences = list(filter(None, sentences))  # drop empty lines
    logging.info('query pre processing...')
    processed_query = sentence_pre_processing(query)
    logging.info('sentences pre processing...')
    processed_sentences = np.asarray(list(map(sentence_pre_processing, sentences)))
    logging.info('loading vectors...')
    word_vectors = load_vectors(WORD_VECTORS_FILE_PATH)
    logging.info('converting query to average vectors...')
    query_avg_vector = np.asarray([get_average_vector(processed_query, word_vectors), ])
    logging.info('converting sentences to average vectors...')
    sentences_avg_vectors = get_average_vectors_from_array(processed_sentences, word_vectors)
    logging.info('predicting sentence of query...')
    predict_vector = model.predict(query_avg_vector)
    logging.info('finding the most similar sentence...')
    cosine_similarity_values = []
    max_val = None
    max_index = 0
    for i, sentence_vector in enumerate(sentences_avg_vectors, 0):
        sentence_vector_2d = np.asarray([sentence_vector, ])  # numpy array with one vector (vector length is 300)
        val = cosine_similarity(predict_vector, sentence_vector_2d)[0][0]  # the highest value is of most similar vector
        cosine_similarity_values.append(val)
        if max_val is None:  # for the first time
            max_val = val
        elif val > max_val:
            max_val = val
            max_index = i
    most_similar_sentence = sentences[max_index]  # most similar sentence
    print(f'"{most_similar_sentence}" sentence is similar to "{query}" query in {max_val} score')
    # save most similar sentence to file
    most_similar_sentence_file_path = os.path.dirname(user_args.query)  # get container directory from file path
    if not most_similar_sentence_file_path:
        most_similar_sentence_file_path = os.curdir
    most_similar_sentence_file_path = os.path.join(most_similar_sentence_file_path, MOST_SIMILAR_FILE_NAME)
    logging.info(f'saving the most similar sentence to {most_similar_sentence_file_path} file...')
    with open(most_similar_sentence_file_path, 'w') as most_similar_sentence_file:
        most_similar_sentence_file.write(most_similar_sentence)
    logging.info('test task done!')


def sentence_pre_processing(raw_sentence):
    """
    Pre processing step on a sentence includes lower case, remove punctuation, remove stop words and filter alphanumeric
    :param raw_sentence: a sentence from raw data
    :return: numpy array of processed sentence
    """
    words = np.asarray(word_tokenize(raw_sentence.lower()))  # lower case and tokenization
    punctuation_removed = map(remove_punctuation, words)  # remove punctuation
    stopwords_filtered = filter(lambda word: word not in ALL_STOPWORDS, punctuation_removed)  # stop word removal
    return np.asarray(list(filter(is_alphanumeric, stopwords_filtered)))  # remove non-alphanumeric words


# mapping task to function
TASKS = {
    'train': train_task_handler,
    'test': test_task_handler
}

ALL_STOPWORDS = stopwords.words('english')


def main():
    """
    Entry point of the script
    """
    args = get_user_cli_args()
    task_handler = TASKS[args.task]  # get the task handler function from global dictionary mapper
    task_handler(args)


if __name__ == '__main__':
    main()
