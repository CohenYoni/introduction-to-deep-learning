"""
Developers:
Yoni Cohen
May Hagbi
"""

from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Activation, Dropout
from keras.layers import LSTM, TimeDistributed
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from nltk.tokenize import word_tokenize
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import logging
import pickle
import sys
import cv2
import os


PROD = False
CHECKPOINTS_DIR_PATH = 'checkpoints'
TRAIN_DATA_CHECKPOINT_NAME = 'train_data.npy'
TRAIN_LABELS_CHECKPOINT_NAME = 'train_labels.npy'
TEST_DATA_CHECKPOINT_NAME = 'test_data.npy'
TEST_LABELS_CHECKPOINT_NAME = 'test_labels.npy'
TRAIN_DATA_CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR_PATH, TRAIN_DATA_CHECKPOINT_NAME)
TRAIN_LABELS_CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR_PATH, TRAIN_LABELS_CHECKPOINT_NAME)
TEST_DATA_CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR_PATH, TEST_DATA_CHECKPOINT_NAME)
TEST_LABELS_CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR_PATH, TEST_LABELS_CHECKPOINT_NAME)

DEFAULT_MODEL2_PATH = 'regression_model.h5'
DEFAULT_MODEL3_PATH = 'image_model.h5'
DEFAULT_MODEL4_PATH = 'text_model.h5'
DEFAULT_WORD_VECTORS_FILE_PATH = 'wiki-news-300d-1M.vec'
BBC_DATASET_FILE_PATH = 'BBC_news_dataset.csv'
RNN_GRAPH_PATH = 'training_rnn_history_graph.png'
MODEL4_SEQUENCE_LENGTH = 3
MODEL4_NUM_OF_UNITS = MODEL4_SEQUENCE_LENGTH
WORD_VECTOR_LENGTH = 300
TEST_PORTION_SIZE = 0.2


logging.basicConfig(stream=sys.stdout, format='%(asctime)s | %(message)s', level=logging.INFO)


def get_user_cli_args():
    """
    Get I/O arguments from user passed from command line.
    :return: user arguments parsed by ArgumentParser
    """
    parser = argparse.ArgumentParser(description='Generate headline for an image')
    parser.add_argument('--task', choices=TASKS.keys(), help='/'.join(TASKS.keys()), required=True)
    parser.add_argument('--image', help='image file global path')
    parser.add_argument('--model2', help='regression model file global path', default=DEFAULT_MODEL2_PATH)
    parser.add_argument('--model3', help='CNN model file global path', default=DEFAULT_MODEL3_PATH)
    parser.add_argument('--model4', help='RNN model file global path', default=DEFAULT_MODEL4_PATH)
    parser.add_argument('--wordvec', help='global path of FastText wv file', default=DEFAULT_WORD_VECTORS_FILE_PATH)
    return parser.parse_args()


def read_bbc_dataset(dataset_file_path):
    """
    Read bbc dataset from a csv file
    :param dataset_file_path: an absolute path for dataset file
    :return: a dictionary of numpy arrays contains tags (query) and description (sentence)
    """
    dataset = pd.read_csv(dataset_file_path)
    dataset.fillna('', inplace=True)
    return {
        'description': np.array(dataset['description']),
        'tags': np.array(dataset['tags'])
    }


def get_cifar_index_to_word_label_mapper(dictionary_file_path):
    """
    Get index-to-word mapper of cifar100 dataset
    :param dictionary_file_path: cifar100 cifar100_meta binary file
    :return: list of words, index i maps to word label #i
    """
    with open(dictionary_file_path, 'rb') as fp:
        label_dictionary = pickle.load(fp, encoding='utf-8')
    return label_dictionary['fine_label_names']


def load_fast_text_vectors(vectors_file_path):
    """
    Load word embedding vectors from a file
    :param vectors_file_path: an absolute path of the file
    :return: numpy array of vectors
    """
    pickle_file_path = f'{vectors_file_path}.pickle'
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as fp:
            data = pickle.load(fp)
    else:
        with open(vectors_file_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fp:
            length, dimension = map(int, fp.readline().split())  # read header
            logging.info(f'{vectors_file_path}: length={length}, dimension={dimension}')
            data = {}
            for line in fp:
                tokens = line.rstrip().split(' ')
                data[tokens[0]] = np.array(tokens[1:], dtype='float32')
        with open(pickle_file_path, 'wb') as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return data


def pixel_treatment(data):
    """
    normalize the data
    :param data: numpuy array of data
    :return: normalized data
    """
    return np.array(data, dtype="float") / 255.0


def remove_bbc_sentences_without_tags(dataset):
    """
    Filter out missing data
    :param dataset: dataset from csv
    :return: correct dataset
    """
    filter_array = dataset['tags'] != ''
    dataset['description'] = dataset['description'][filter_array]
    dataset['tags'] = dataset['tags'][filter_array]
    return dataset


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


def sentence_words_to_vectors(sentence, word_vectors, ctr=None):
    """
    Represent every word W in your sentence as its word vector V(W)
    :param sentence: numpy array of words
    :param word_vectors: FastText
    :param ctr: counter of missing words
    :return: sentence of vectors
    """
    sentence_as_vectors = []
    for word in sentence:
        try:
            sentence_as_vectors.append(word_vectors[word])
        except KeyError:
            logging.info(f'there is no vector for {word}, ignored!')
            sentence_as_vectors.append(np.zeros(300))
            if ctr:
                ctr['missing'] += 1
    return np.array(sentence_as_vectors)


def create_sequence_and_label(sentences):
    """
    Generate sequences and labels from the sentences.
    The sequences will be consecutive L-word sequences from the sentences for which a next word exists
    The label will be the next word
    :param sentences: numpy array of sentences as vectors
    :return: sequences and labels
    """
    sequences, labels = [], []
    seq_len = MODEL4_SEQUENCE_LENGTH
    for sentence in sentences:
        curr_sequence = np.array([sentence[i: i + seq_len] for i in range(len(sentence) - seq_len)])
        curr_label = np.array([sentence[i] for i in range(seq_len, len(sentence))])
        sequences.extend(curr_sequence)
        labels.extend(curr_label)
    return np.array(sequences), np.array(labels)


def rnn_pre_processing(dataset):
    """
    Sentence splitting, tokenization
    :param dataset: bbc dataset
    :return: numpy array of words
    """
    logging.info('remove sentences without tags...')
    dataset = remove_bbc_sentences_without_tags(dataset)
    logging.info('sentence splitting...')
    sentences = dataset['description']
    logging.info('sentences tokenization....')
    return np.array([np.array(word_tokenize(s.lower())) for s in sentences])


def build_rnn_model():
    """
    Build RNN model
    :return: keras model
    """
    model = Sequential()
    input_shape = (MODEL4_SEQUENCE_LENGTH, WORD_VECTOR_LENGTH)
    model.add(LSTM(MODEL4_NUM_OF_UNITS, return_sequences=True, activation="tanh", input_shape=input_shape))
    model.add(LSTM(MODEL4_NUM_OF_UNITS, return_sequences=False, activation="tanh"))
    model.add(Dense(300))
    # model.add(TimeDistributed(Dense(300)))
    return model


def rnn_training(train_data, train_labels, test_data, test_labels, model_file_path):
    # model definition

    train_data = train_data[:200]
    train_labels = train_labels[:200]
    test_data = test_data[:100]
    test_labels = test_labels[:100]

    logging.info('defining rnn model...')
    model = build_rnn_model()
    # model compilation
    logging.info('compiling rnn model...')
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.01), metrics=['accuracy'])
    # summarize model
    logging.info('summarize rnn model...')
    model.summary()
    # model fitting
    logging.info('training rnn...')
    training_history = model.fit(train_data, train_labels, epochs=20, batch_size=50, validation_split=0.1)
    # model evaluation
    logging.info('evaluating rnn...')
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)
    print(f'evaluating scores: loss = {test_loss}, accuracy = {test_accuracy}')
    # save model to json and h5 files
    logging.info('save rnn model...')
    save_model_to_h5(model, model_file_path)
    logging.info('creating training rnn graph...')
    create_training_graph(training_history, RNN_GRAPH_PATH,
                          test_history={'test_loss': test_loss, 'test_accuracy': test_accuracy}, with_validation=True)


def train_task_handler(user_args):
    """
    Run training task
    :param user_args: user cli arguments
    """
    assert user_args.wordvec, 'word vectors file path is missing'
    assert user_args.model4, 'text model file path is missing'
    if PROD or not checkpoint_files_exists():
        logging.info('loading bbc dataset...')
        bbc_dataset = read_bbc_dataset(BBC_DATASET_FILE_PATH)
        logging.info('performing rnn pre processing...')
        sentences_as_words = rnn_pre_processing(bbc_dataset)
        del bbc_dataset
        logging.info('loading vectors...')
        word_vectors = load_fast_text_vectors(user_args.wordvec)
        logging.info('representing sentence words as vector...')
        ctr = {'missing': 0}
        sentences_as_vectors = np.array([sentence_words_to_vectors(s, word_vectors, ctr) for s in sentences_as_words])
        del word_vectors, sentences_as_words
        logging.info(f'{ctr["missing"]} words were ignored!')
        sequences, labels = create_sequence_and_label(sentences_as_vectors)
        del sentences_as_vectors
        logging.info(f'splitting to {(1 - TEST_PORTION_SIZE)*100}% train / {TEST_PORTION_SIZE * 100}% test...')
        train_data, test_data, train_labels, test_labels = train_test_split(sequences, labels,
                                                                            test_size=TEST_PORTION_SIZE, shuffle=True,
                                                                            random_state=np.random.randint(1, high=100))
        del sequences, labels
        os.makedirs(CHECKPOINTS_DIR_PATH, exist_ok=True)
        np.save(TRAIN_DATA_CHECKPOINT_PATH, train_data)
        np.save(TRAIN_LABELS_CHECKPOINT_PATH, train_labels)
        np.save(TEST_DATA_CHECKPOINT_PATH, test_data)
        np.save(TEST_LABELS_CHECKPOINT_PATH, test_labels)
    else:
        train_data = np.load(TRAIN_DATA_CHECKPOINT_PATH, allow_pickle=True)
        train_labels = np.load(TRAIN_LABELS_CHECKPOINT_PATH, allow_pickle=True)
        test_data = np.load(TEST_DATA_CHECKPOINT_PATH, allow_pickle=True)
        test_labels = np.load(TEST_LABELS_CHECKPOINT_PATH, allow_pickle=True)
    logging.info('start training rnn..')
    rnn_training(train_data, train_labels, test_data, test_labels, user_args.model4)
    logging.info('train rnn task done!')


def read_image(image_path):
    """
    read image and convert it to 32*32*3 pixels
    :param image_path: an image path
    :return: numpy array of 32*32*3 pixels image
    """
    image = cv2.imread(image_path)
    return cv2.resize(image, (32, 32))  # to get images sized 3072


def find_similar_word(vector, word_vectors_dict):
    """
    find the most similar word from dictionary
    :param vector: word as vector
    :param word_vectors_dict: dictionary of word->vectors
    :return: most similar word
    """
    max_val_idx = find_similar_idx(vector, np.array(list(word_vectors_dict.values())))
    return list(word_vectors_dict.keys())[int(max_val_idx)]


def find_similar_idx(vector, vectors_array):
    """
    find the index of most similar vector
    :param vector: vector to check most similarity
    :param vectors_array: array of vectors
    :return: index
    """
    similarities = cosine_similarity(vector, vectors_array)
    return np.argmax(similarities.reshape(similarities.shape[1]))


def test_task_handler(user_args):
    """
    Run testing task
    :param user_args: user cli arguments
    """
    assert user_args.image, 'image file path is missing'
    assert user_args.wordvec, 'word vectors file path is missing'
    assert user_args.model2, 'regression model file path is missing'
    assert user_args.model3, 'image model file path is missing'
    assert user_args.model4, 'text model file path is missing'
    # loading saved models
    logging.info('loading regression model...')
    regression_model = load_model(user_args.model2)
    regression_model.summary()
    logging.info('loading image model...')
    image_cnn_model = load_model(user_args.model3)
    image_cnn_model.summary()
    logging.info('loading text model...')
    text_rnn_model = load_model(user_args.model4)
    text_rnn_model.summary()
    # read user image to perform prediction (image classification)
    logging.info(f'reading {user_args.image}...')
    image_tensor = read_image(user_args.image)
    # pre precessing
    image_tensor = pixel_treatment(image_tensor.reshape(1, 32, 32, 3))
    # perform prediction
    logging.info('predicting...')
    predict_vector = image_cnn_model.predict(image_tensor)
    # loading fasttext vectors
    logging.info('loading vectors...')
    word_vectors = load_fast_text_vectors(user_args.wordvec)
    # find the most similar vectors from fasttext
    logging.info('calculating cosine similarity...')
    word1 = find_similar_word(predict_vector, word_vectors)
    rnn_input1 = np.array([word_vectors[word1], ] * MODEL4_SEQUENCE_LENGTH)
    rnn_input1 = rnn_input1.reshape((1, MODEL4_SEQUENCE_LENGTH, WORD_VECTOR_LENGTH))
    word2_vector = text_rnn_model.predict(rnn_input1)
    rnn_input2 = np.array([word2_vector, ] * MODEL4_SEQUENCE_LENGTH)
    rnn_input2 = rnn_input2.reshape((1, MODEL4_SEQUENCE_LENGTH, WORD_VECTOR_LENGTH))
    word3_vector = text_rnn_model.predict(rnn_input2)
    word2 = find_similar_word(word2_vector, word_vectors)
    word3 = find_similar_word(word3_vector, word_vectors)
    headline = (word1, word2, word3)
    print(f'headline = {headline}')
    headline_as_avg_vector = np.array([word_vectors[word] for word in headline]).mean(axis=0)
    alternative_headline_vector = regression_model.predict(headline_as_avg_vector.reshape(1, 300))
    bbc_dataset = read_bbc_dataset(BBC_DATASET_FILE_PATH)
    ctr = {'missing': 0}
    sentences_as_words = rnn_pre_processing(bbc_dataset)
    del bbc_dataset
    sentences_as_vectors = np.array([sentence_words_to_vectors(s, word_vectors, ctr) for s in sentences_as_words])
    sentences_as_avg_vectors = np.array([s.mean(axis=0) for s in sentences_as_vectors])
    alternative_headline_idx = find_similar_idx(alternative_headline_vector, sentences_as_avg_vectors)
    alternative_headline = ' '.join(sentences_as_words[alternative_headline_idx])
    print(f'alternative headline = {alternative_headline}')
    logging.info('test task done!')


def checkpoint_files_exists():
    """
    check if the pre precessed data and test exists
    """
    train_data_exists = os.path.exists(TRAIN_DATA_CHECKPOINT_PATH)
    train_labels_exists = os.path.exists(TRAIN_LABELS_CHECKPOINT_PATH)
    test_data_exists = os.path.exists(TEST_DATA_CHECKPOINT_PATH)
    test_labels_exists = os.path.exists(TEST_LABELS_CHECKPOINT_PATH)
    return train_data_exists and train_labels_exists and test_data_exists and test_labels_exists


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
