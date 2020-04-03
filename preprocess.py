"""
Developers:
Yoni Cohen
May Hagbi
"""

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer
from argparse import ArgumentParser
from nltk.corpus import stopwords
import wordcloud
import string
import os
import re


def is_alphabetic(word_str):
    return re.match(r'^[a-zA-Z]+$', word_str) is not None


def is_alphanumeric(word_str):
    return re.match(r'^[a-zA-Z0-9]*[a-zA-Z][a-zA-Z0-9]*$', word_str) is not None


def remove_punctuation(word_str):
    return re.sub(rf'[{string.punctuation}]', '', word_str)


def read_data_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as input_file:
        file_content = input_file.read()
    return file_content


def pre_processing(raw_data):
    stemmer = PorterStemmer()
    raw_sentences = sent_tokenize(raw_data)
    stemmed_sentences = []
    for sentence in raw_sentences:
        punctuation_removed = map(remove_punctuation, word_tokenize(sentence.lower()))
        stopwords_filtered = filter(lambda word: word not in stopwords.words(), punctuation_removed)
        stemmed = list(filter(is_alphanumeric, map(stemmer.stem, stopwords_filtered)))
        stemmed_sentences.append(stemmed)
    return stemmed_sentences


def create_word_cloud(stemmed_sentences, img_path):
    words_counter_dict = {}
    for sentence in stemmed_sentences:
        for word in sentence:
            words_counter_dict[word] = words_counter_dict.get(word, 0) + 1
    word_cloud = wordcloud.WordCloud(
        background_color="#101010",
        width=1080,
        height=720,
        max_words=20
    )
    word_cloud.generate_from_frequencies(words_counter_dict)
    word_cloud.to_file(img_path)


def map_words_to_index(stemmed_sentences):
    words_list = sorted(list(set([word for sentence in stemmed_sentences for word in sentence])))
    mapped_words_dict = {}
    for index, word in enumerate(words_list):
        mapped_words_dict[word] = index
    return mapped_words_dict


def create_1_hot_representation_matrix(stemmed_sentences, mapped_words_dict):
    matrix_1_hot_representation = []
    vector_len = len(mapped_words_dict)
    for sentence in stemmed_sentences:
        sentence_1_hot_representation = []
        for word in sentence:
            word_1_hot_representation = [0, ] * vector_len
            word_1_hot_representation[mapped_words_dict[word]] = 1
            sentence_1_hot_representation.append(word_1_hot_representation)
        matrix_1_hot_representation.append(sentence_1_hot_representation)
    return matrix_1_hot_representation


def write_matrix_to_file(matrix, file_path):
    with open(file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(str(matrix))


def main():
    args_parser = ArgumentParser()
    args_parser.add_argument('input_file_path', help='Absolute path for input data file.')
    args = args_parser.parse_args()
    file_name_prefix = os.path.basename(os.path.splitext(args.input_file_path)[0])
    file_path = os.path.dirname(args.input_file_path)
    if not file_path:
        file_path = os.curdir
    file_data = read_data_from_file(args.input_file_path)
    stemmed_sentences = pre_processing(file_data)
    create_word_cloud(stemmed_sentences, os.path.join(file_path, file_name_prefix + '_cloud.png'))
    mapped_words_dict = map_words_to_index(stemmed_sentences)
    output_file_path = os.path.join(file_path, file_name_prefix + '1hot.txt')
    write_matrix_to_file(create_1_hot_representation_matrix(stemmed_sentences, mapped_words_dict), output_file_path)


if __name__ == '__main__':
    main()
