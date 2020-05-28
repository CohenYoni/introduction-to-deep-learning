"""
Developers:
Yoni Cohen
May Hagbi
"""

from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Activation, Dropout
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
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
GRAPH_PATH = 'training_history_graph.png'
DEFAULT_MODEL_PATH = 'image_model.h5'
TOP_SIMILARITIES = 3
TEST_IMAGE_PATH = 'train.jpg'

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


def map_words_labels_to_word_vector(words_labels, word_vectors):
    """
    map every label to word vector
    :param words_labels: numpy array of word(s) labels
    :param word_vectors: fast text words vectors
    :return: numpy array of word vectors
    """

    def get_avg_wv_of_label(label_with_several_words):
        """
        calculate vector average of label with several word
        :param label_with_several_words: label with several word separate by '_'
        :return: average vector
        """
        labels = np.array(label_with_several_words.split('_'))
        return np.array([word_vectors[word] for word in labels]).mean(axis=0)

    logging.info('converts words to vectors...')
    return np.array([word_vectors[word] if '_' not in word else get_avg_wv_of_label(word) for word in words_labels])


def pre_processing(dataset, word_vectors):
    """
    performs pre processing of data and labels
    :param dataset: data and labels
    :param word_vectors: fast text words vectors
    :return: pre processed data and labels
    """
    data, labels = dataset
    data = pixel_treatment(data)
    labels = map_label_index_to_word(labels)
    labels = map_words_labels_to_word_vector(labels, word_vectors)
    return data, labels


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


def training(train_data, train_labels, test_data, test_labels, model_file_path):
    """
    Training the model
    Stages: model definition, model compilation, model fitting (training), model evaluation
    :param train_data: numpy array of cifar100 images
    :param train_labels: numpy array of labels word vectors
    :param test_data: numpy array of cifar100 images
    :param test_labels: numpy array of labels word vectors
    :param model_file_path: an absolute path for file who saves the model structure and weights
    """

    # TODO: REMOVE!!!
    # train_data, train_labels, test_data, test_labels = train_data[:2], train_labels[:2], test_data[:2], test_labels[:2]
    # TODO: END

    width, height, depth = 32, 32, 3
    chan_dim = -1  # channels_last format
    input_shape = (width, height, depth)
    padding = 'same'
    # model definition
    logging.info('defining the model...')
    model = Sequential()
    # layer #1
    model.add(Conv2D(32, (3, 3), padding=padding, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # layer #2-3
    for _ in range(2):
        model.add(Conv2D(32, (3, 3), padding=padding))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
    # final layer
    model.add(Flatten())
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chan_dim))
    # model.add(Dropout(0.5))
    # model compilation
    logging.info('compiling the model...')
    # mean_squared_error
    # categorical_crossentropy
    model.compile(optimizer=Adam(lr=0.01), loss='mean_squared_error', metrics=['accuracy'])
    # summarize model
    logging.info('summarize model...')
    model.summary()
    # model fitting
    logging.info('training...')
    training_history = model.fit(train_data, train_labels, epochs=100, batch_size=10)
    # model evaluation
    logging.info('evaluating...')
    test_loss, test_accuracy = model.evaluate(test_data, test_labels, batch_size=50)
    print(f'evaluating scores: loss = {test_loss}, accuracy = {test_accuracy}')
    logging.info('creating training graph...')
    create_training_graph(training_history, GRAPH_PATH,
                          test_history={'test_loss': test_loss, 'test_accuracy': test_accuracy})
    # save model to json and h5 files
    logging.info('save model...')
    save_model_to_h5(model, model_file_path)


def train_task_handler(user_args):
    """
    Run training task
    :param user_args: user cli arguments
    """
    logging.info('loading dataset...')
    if PROD:
        train, test = get_train_and_test_data()
        logging.info('loading vectors...')
        word_vectors = load_fast_text_vectors(WORD_VECTORS_FILE_PATH)
        logging.info('performing train pre processing...')
        train_data, train_labels = pre_processing(train, word_vectors)
        np.save('train_data.npy', train_data)
        np.save('train_labels.npy', train_labels)
        logging.info('performing test pre processing...')
        test_data, test_labels = pre_processing(test, word_vectors)
        np.save('test_data.npy', test_data)
        np.save('test_labels.npy', test_labels)
    else:
        train_data = np.load('train_data.npy')
        train_labels = np.load('train_labels.npy')
        test_data = np.load('test_data.npy')
        test_labels = np.load('test_labels.npy')
    logging.info('start training..')
    training(train_data, train_labels, test_data, test_labels, user_args.model)
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
    """
    Run testing task
    :param user_args: user cli arguments
    """
    logging.info('loading model...')
    model = load_model(user_args.model)
    model.summary()
    logging.info(f'reading {TEST_IMAGE_PATH}...')
    image_tensor = read_image(TEST_IMAGE_PATH)
    image_tensor = pixel_treatment(image_tensor.reshape(1, 32, 32, 3))
    # TODO: REMOVE!!!
    # train, _ = get_train_and_test_data()
    # image_tensor = pixel_treatment(np.array([train[0][0], ]))
    # TODO: END
    logging.info('predicting...')
    predict_vector = model.predict(image_tensor)
    logging.info('loading vectors...')
    word_vectors = load_fast_text_vectors(WORD_VECTORS_FILE_PATH)
    logging.info('calculating cosine similarity...')
    cosine_similarity_values = []
    # items = np.array(list(word_vectors.items()))
    # wv is a tuple: (word, 300D vector)
    # similarities = np.array([{wv[0]: cosine_similarity(predict_vector, wv[1].reshape(1, 300))} for wv in items])
    similarities = []
    i = 0
    for word, vector in word_vectors.items():
        # if i >= 10000:
        #     break
        similarities.append({'word': word,
                             'val': cosine_similarity(predict_vector, vector.reshape(1, 300))})
        i += 1
    logging.info(f'finding top {TOP_SIMILARITIES}...')
    similarities = sorted(similarities, key=lambda x: x['val'], reverse=True)
    for i in range(TOP_SIMILARITIES):
        print(f'<{similarities[i]["word"]}>', end=' ')
    print('')
    logging.info('test task done!')


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
    PROD = False
    main()
