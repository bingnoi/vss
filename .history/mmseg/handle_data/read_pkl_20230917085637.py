import numpy as np
import pickle

pk = pickle.load(open('datasets/coco/coco_stuff/word_vectors/fasttext.pkl', "rb"))

print(pk.shape)