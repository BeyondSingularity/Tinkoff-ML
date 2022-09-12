import argparse
import pickle
import os
import sys
import random
import hashlib
# import numpy - я так понял, что не могу его использовать,
# т.к. не стандартная библиотека и не указана отдельно


class MyModel():
    def __init__(self):
        self.ngram = {}
        self.seed = 0

    # x_train - list with training data,
    # y_train - answers for training data
    def fit(self, x_train, y_train):
        temp_ngram = {}     # not completed n-gram.

        # training part
        for i in range(len(x_train)):
            word = x_train[i]       # input word
            result = y_train[i]     # right output word

            # every word associates with possible next words
            if word not in temp_ngram:
                temp_ngram[word] = {}

            # counting how many times each word was used
            if result not in temp_ngram[word]:
                temp_ngram[word][result] = 1
            else:
                temp_ngram[word][result] += 1

        # forming completed n-gram with data from temp_ngram
        for word, results in temp_ngram.items():
            n = 0
            for quantity in results.values():
                n += quantity
            next_words_probabilities = []
            for res_word, res_probability in results.items():
                # probability of word Y being after word X is
                # P(how many times word Y was after X) / n,
                # where n - sum of all training tests with word X
                next_words_probabilities.append(
                    (res_word, res_probability / n)
                )
            self.ngram[word] = tuple(next_words_probabilities)

    # x_eval - data for evaluation
    # lenght - length of completed sentence
    def generate(self, x_eval, length):

        # if x_eval wasn't stated, choosing from all available words
        if len(x_eval) == 0:
            x_eval = [random.choice(list(self.ngram.keys()))]

        # logically, all given words should be part of finished sentence
        sentence = x_eval.copy()

        # because my n-gram takes only 1 word as an input
        # (I tried using 2 or more, but results were very boring fr no cap)
        # now - is a word that acts as an input right now, hence the name
        now = x_eval[-1]

        # all x_eval words are already used, so -len(x_eval)
        # max is a safety measure
        for i in range(max(length - len(x_eval), 0)):

            # searching for most suitable word
            next_word, mx = '', -1
            if now not in self.ngram:   # if can't determine for "now" - skip
                now = random.choice(list(self.ngram.keys()))
            for elem in self.ngram[now]:

                # I decided to use seed value to "spice up" results
                # This makes values vary by 0.1 up/down pseudo-randomly
                hsh = hashlib.md5(str.encode(elem[0]))
                spice = int(hsh.hexdigest(), 16) + self.seed
                new_value = elem[1] + (spice % 20000 - 10000) / 10000

                # Taking word with highest spiced associated value
                if new_value > mx:
                    next_word, mx = elem[0], new_value
            # adding new word to the end of the sentence
            sentence.append(next_word)

            # last word becomes input for new one
            now = next_word
        return sentence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process model training parameters.',
        usage='"Training" of the model',
    )
    parser.add_argument(
        "--input-dir",
        dest="path_to_input_dir",
        help="The path to the directory where the\
            collection of documents is located.",
    )
    parser.add_argument(
        "--model",
        dest="path_to_model",
        required=True,
        help="Path to the file in which the model is saved.",
    )
    args = parser.parse_args()

    texts = []  # all unparced texts will be here
    if not args.path_to_input_dir:
        s = None
        for line in sys.stdin:
            texts = ["Sorry, I have no idea how to read args\
                and stdin simultaneously"]
    else:
        # collecting all path to files with data
        for (dirpath, dirnames, filenames) in os.walk(args.path_to_input_dir):
            for file in filenames:
                filepath = dirpath + '\\' + file
                # opening all files
                with open(filepath, encoding='utf-8') as f:
                    text = f.read()
                # collecting all data in one list (in lowercase)
                texts.append(text.lower())

    processed_texts = []    # self-explanatory
    for text in texts:
        processed_text = []
        word = ''

        # non-alphabetic symbol added to remove redundancy later
        for i in text + '.':
            if i.isalpha():
                word += i
            elif word != '':
                processed_text.append(word)
                word = ''
        processed_texts.append(processed_text)

    # explained in MyModel.fit function
    x_train = []
    y_train = []
    for text in processed_texts:
        for i in range(len(text) - 1):
            x_train.append(text[i])
            y_train.append(text[i + 1])

    # creating class instance
    model = MyModel()

    # training it
    model.fit(x_train, y_train)

    # saving all important data
    with open(args.path_to_model, 'wb') as f:
        pickle.dump(model.ngram, f)
