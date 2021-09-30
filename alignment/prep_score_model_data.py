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
    "Example", "identifier review_sentence rebuttal_sentence score".split())


def preprocess_sentences(sentences):
  return zip(*[(preprocess(sentence["text"]), sentence["text"])
               for sentence in sentences])


def identifier_maker(review_id, review_index, rebuttal_index):
  return (review_id, review_index, rebuttal_index)
  #return "{0}_{1}_{2}".format(review_id, review_index, rebuttal_index)

def unpack_identifier(identifier):
  review_id, review_index, rebuttal_index = identifier.rsplit("_", 2)
  return review_id, int(review_index), int(rebuttal_index)


def make_pair_examples(review_id, review_sentences, rebuttal_sentences):
  corpus, review_texts = preprocess_sentences(review_sentences)
  preprocessed_queries, query_texts = preprocess_sentences(rebuttal_sentences)
  model = rank_bm25.BM25Okapi(corpus)

  examples = []
  identifiers = []

  for rebuttal_index, preprocessed_query in enumerate(preprocessed_queries):
    scores = model.get_scores(preprocessed_query)
    assert len(scores) == len(review_sentences)
    for review_index, score in enumerate(scores):
      identifier = identifier_maker(review_id, review_index, rebuttal_index)
      examples.append(
          Example(identifier, review_texts[review_index],
          query_texts[rebuttal_index], score))
      identifiers.append(identifier)
  return examples, identifiers


def make_text_example(index, example):
  return json.dumps({
      "index":
          index,
      "text":
          " [SEP] ".join([example.review_sentence, example.rebuttal_sentence]),
      "label":
          example.score
  })


def get_vocab_from_examples(examples):
  tokens = set()
  for example in examples:
    for sentence in [example.review_sentence, example.rebuttal_sentence]:
      tokens.update(set(nltk.word_tokenize(sentence)))
  return tokens


def get_matches_from_obj(obj):
  matches = collections.defaultdict(lambda: collections.defaultdict(list))
  review_id = obj["metadata"]["review_id"]
  for j, rebuttal_sentence in enumerate(obj["rebuttal_sentences"]):
    alignment_type, maybe_context = rebuttal_sentence["alignment"]
    if alignment_type == "context_sentences":
      for aligned_sentence in set(maybe_context):
        if aligned_sentence < len(obj["review_sentences"]):
          matches[review_id][j].append(aligned_sentence)
  return matches


def main():

  args = parser.parse_args()
  overall_tokens = set()
  example_identifiers = []
  true_matches = {}
  example_offset = 0

  for subset in "train dev test".split():

    all_filenames = glob.glob(args.input_dir + "/" + subset + "/*")
    chunk_size = int(len(all_filenames) / 30)

    for batch_index, offset in tqdm.tqdm(
        enumerate(range(0, len(all_filenames), chunk_size))):
      subset_examples = []
      filenames = all_filenames[offset:offset + chunk_size]
      for filename in filenames:
        with open(filename, 'r') as f:
          obj = json.load(f)
          true_matches.update(get_matches_from_obj(obj))
          review_id = obj["metadata"]["review_id"]
          examples, identifiers = make_pair_examples(review_id,
                                                     obj["review_sentences"],
                                                     obj["rebuttal_sentences"])
          subset_examples += examples
          example_identifiers += identifiers
      overall_tokens.update(get_vocab_from_examples(subset_examples))

      with open(
          "score_model_input/" + subset + "_{0}.jsonl".format(batch_index),
          'w') as f:
        f.write("\n".join([
            make_text_example(example_identifiers.index(example.identifier),
                              example) for example in subset_examples
        ]))

  with open("score_model_input/metadata.json", 'w') as f:
    json.dump(
        {
            "example_identifiers": example_identifiers,
            "true_matches": true_matches
        }, f)


if __name__ == "__main__":
  main()
