import argparse
import collections
import glob
import json
import nltk
import tqdm

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import rank_bm25

parser = argparse.ArgumentParser(description="Prepare input for IR score model")
parser.add_argument('-i',
                    '--input_dir',
                    default="../data_prep/final_dataset",
                    type=str,
                    help='path to dataset directory')

STEMMER = PorterStemmer()
STOPWORDS = stopwords.words('english')


def preprocess(sentence):
  return [
      STEMMER.stem(word).lower()
      for word in nltk.word_tokenize(sentence)
      if word.lower() not in STOPWORDS
  ]


Example = collections.namedtuple(
    "Example", "review_sentence rebuttal_sentence score".split())


def preprocess_sentences(sentences):
  return zip(*[(preprocess(sentence["text"]), sentence["text"])
               for sentence in sentences])


def make_pair_examples(review_sentences, rebuttal_sentences):
  corpus, review_texts = preprocess_sentences(review_sentences)
  preprocessed_queries, query_texts = preprocess_sentences(rebuttal_sentences)
  model = rank_bm25.BM25Okapi(corpus)

  examples = []

  for i, preprocessed_query in enumerate(preprocessed_queries):
    scores = model.get_scores(preprocessed_query)
    assert len(scores) == len(review_sentences)
    for j, score in enumerate(scores):
      examples.append(Example(review_texts[j], query_texts[i], score))

  return examples


def make_text_example(index, example):
  return json.dumps({
      "index":
          index,
      "text":
          " [SEP] ".join([example.review_sentence, example.rebuttal_sentence]),
      "label":
          example.score
  })


def main():

  args = parser.parse_args()



  for subset in "train dev test".split():

    all_filenames = glob.glob(args.input_dir + "/" + subset + "/*")
    chunk_size = int(len(all_filenames)/9)

    for i, offset in tqdm.tqdm(enumerate(range(0, len(all_filenames),
    chunk_size))):
      subset_examples = []
      filenames = all_filenames[offset:offset+chunk_size]
      print(len(filenames))
      for filename in filenames:
        with open(filename, 'r') as f:
          obj = json.load(f)
          subset_examples += make_pair_examples(obj["review_sentences"],
                                                obj["rebuttal_sentences"])

      with open("score_model_input/" + subset + "_{0}.jsonl".format(i), 'w') as f:
        f.write("\n".join([
            make_text_example(i, example)
            for i, example in enumerate(subset_examples)
        ]))


if __name__ == "__main__":
  main()
