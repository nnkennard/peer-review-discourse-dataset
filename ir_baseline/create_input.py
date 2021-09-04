import argparse
import collections
import glob
import json
import math
import os
import random
import torch
import tqdm

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import rank_bm25

parser = argparse.ArgumentParser(
    description="Convert the filtered database entries into a cleaned dataset.")
parser.add_argument('-i',
                    '--input_dir',
                    default="../data_prep/final_dataset",
                    type=str,
                    help='path to dataset directory')
parser.add_argument('-o',
                    '--output_dir',
                    default="rank_model_input/",
                    type=str,
                    help='path to output directory')



STEMMER = PorterStemmer()
STOPWORDS = stopwords.words('english')


def preprocess(sentence_tokens):
  return [
      STEMMER.stem(word).lower()
      for word in sentence_tokens
      if word.lower() not in STOPWORDS
  ]


def sigmoid(x):
  return 1 / (1 + math.exp(-1 * x))


Example = collections.namedtuple("Example",
    "review_sentence rebuttal_sentence score".split())

def build_example(index, example):
  return json.dumps({
    "index": index,
    "text": " [SEP] ".join([example.review_sentence, example.rebuttal_sentence]),
    "label": example.score,
  })


def get_review_rebuttal_sentences(filename):
  with open(filename, 'r') as f:
    obj = json.load(f)
    review_sentences = [(sent["text"], preprocess(sent["text"]))
                        for sent in obj["review_sentences"]]
    rebuttal_sentences = [(sent["text"], preprocess(sent["text"]))
                          for sent in obj["rebuttal_sentences"]]
  return review_sentences, rebuttal_sentences


def main():

  args = parser.parse_args()

  if not os.path.exists(args.output_dir +"/"):
    os.makedirs(args.output_dir)

  for subset in "train dev test".split():
    examples = []
    for filename in tqdm.tqdm(glob.glob(args.input_dir + "/"+subset+"/*")):
      review_sentences, rebuttal_sentences = get_review_rebuttal_sentences(
          filename)
      model = rank_bm25.BM25Okapi([preprocessed_sentence for (_,
      preprocessed_sentence) in review_sentences])
      for j, (rebuttal_sentence, preprocessed_query) in enumerate(rebuttal_sentences):
        scores = model.get_scores(preprocessed_query)
        assert len(scores) == len(review_sentences)
        for (review_sentence, _), score in zip(review_sentences, scores):
          examples.append(Example(review_sentence, rebuttal_sentence, score))

    
    print(len(examples))
    examples = random.sample(examples, int(len(examples)/10))
    print(len(examples))
    with open(args.output_dir + "/" + subset + ".jsonl" , 'w') as f:
        f.write("\n".join([build_example(i, e) for i, e in enumerate(examples)]))

if __name__ == "__main__":
  main()
