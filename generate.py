import argparse
import pickle
import os
import sys
import random
import hashlib

# exporting model class from train.py
from train import MyModel


parser = argparse.ArgumentParser(
    description='Process parameters to generate text.',
    usage='Generate text based on user input.',
)
parser.add_argument(
    "--model",
    dest="path_to_model",
    required=True,
    help="Path to the file in which the model is saved.",
)
parser.add_argument(
    "--prefix",
    dest="prefix",
    help='The beginning of a sentence (one or more words).',
    nargs='*',
    default=[],
)
parser.add_argument(
    "--length",
    dest="length",
    required=True,
    help="The length of the generated sequence.",
    type=int,
)
args = parser.parse_args()

# creating new instance of MyModel
model = MyModel()

# loading model data
with open(args.path_to_model, 'rb') as f:
    model.ngram = pickle.load(f)

# selecting seed (i think not the best way to do so, but it works)
model.seed = random.randint(0, 1e9)

# [i.lower() for i in args.prefix] turns all user input in lowercase
sentence = model.generate([i.lower() for i in args.prefix], args.length)

print(' '.join(sentence))
