import argparse
import collections
import glob
import json
import os
import random
import sys
import tqdm

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from transformers import BertTokenizer
import rank_bm25

STEMMER = PorterStemmer()
STOPWORDS = stopwords.words('english')

# === IR utils


def preprocess(sentence):
  return [
      STEMMER.stem(word).lower()
      for word in word_tokenize(sentence)
      if word.lower() not in STOPWORDS
  ]


def preprocess_sentences(sentences):
  return zip(*[(preprocess(sentence["text"]), sentence["text"])
               for sentence in sentences])


# === General utils

TRAIN, DEV, TEST = "train dev test".split()
SUBSETS = [TRAIN, DEV, TEST]

Example = collections.namedtuple(
    "Example",
    "overall_index identifier review_sentence rebuttal_sentence both_sentences score label"
    .split())

RelatedPair = collections.namedtuple("RelatedPair",
                                     "review_idx rebuttal_idx".split())


def get_true_related_pairs(rebuttal_sentences):
  true_pairs = []
  for rebuttal_index, rebuttal_sentence in enumerate(rebuttal_sentences):
    align_type, review_indices = rebuttal_sentence["alignment"]
    if align_type == "context_sentences":
      true_pairs += [
          RelatedPair(review_idx=review_index, rebuttal_idx=rebuttal_index)
          for review_index in review_indices
      ]
  return true_pairs


def identifier_maker(review_id, review_index, rebuttal_index):
  return (review_id, review_index, rebuttal_index)


def overall_indexifier():
  index = 0
  while True:
    yield index
    index += 1


TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')


def get_token_vocab(review_sentences, rebuttal_sentences):
  tokens = set()
  for sentence in review_sentences + rebuttal_sentences:
    normal_tokens = word_tokenize(sentence.lower())
    bert_tokens = TOKENIZER.tokenize(sentence)
    tokens.update(
        set(normal_tokens).union(tok for tok in bert_tokens if '#' not in tok))
  return tokens


def make_pair_examples(review_id, review_sentences, rebuttal_sentences,
                       index_generator):
  true_related_pairs = get_true_related_pairs(rebuttal_sentences)
  corpus, review_sentence_texts = preprocess_sentences(review_sentences)
  preprocessed_queries, rebuttal_sentence_texts = preprocess_sentences(
      rebuttal_sentences)
  model = rank_bm25.BM25Okapi(corpus)

  examples = []
  identifiers = []

  example_maps = collections.defaultdict(list)
  for rebuttal_index, preprocessed_query in enumerate(preprocessed_queries):
    scores = model.get_scores(preprocessed_query)
    assert len(scores) == len(review_sentences)
    for review_index, score in enumerate(scores):
      identifier = identifier_maker(review_id, review_index, rebuttal_index)
      label = 1 if RelatedPair(review_index,
                               rebuttal_index) in true_related_pairs else 0
      overall_example_index = next(index_generator)
      review_sentence = review_sentence_texts[review_index]
      rebuttal_sentence = rebuttal_sentence_texts[rebuttal_index]
      both_sentences = review_sentence + " [SEP] " + rebuttal_sentence
      example_maps[label].append(
          Example(overall_example_index, identifier,
                  review_sentence_texts[review_index], both_sentences,
                  rebuttal_sentence_texts[rebuttal_index], score, label))
      identifiers.append((overall_example_index, identifier))
    pos_examples = len(example_maps[1])
    original_neg_examples = len(example_maps[0])
    sampled_neg_examples = random.sample(
        example_maps[0], max(min(len(example_maps[0]), 2 *
        len(example_maps[1])), 3))


    print(collections.Counter(collections.Counter(k.overall_index for k in
    example_maps[1]).values()))


    #print(pos_examples, original_neg_examples, len(sampled_neg_examples))

    examples += example_maps[1]
    examples += sampled_neg_examples
  random.shuffle(examples)


  #print(collections.Counter(x.label for x in examples))

  return examples, identifiers, get_token_vocab(review_sentence_texts,
                                                rebuttal_sentence_texts)


def make_output_filename(output_dir, subset, index):
  return "/".join([output_dir, subset, str(index).zfill(4) + ".jsonl"])


def get_general_examples(input_filename, index_generator):
  with open(input_filename, 'r') as f:
    obj = json.load(f)
    return make_pair_examples(obj["metadata"]["review_id"],
                              obj["review_sentences"],
                              obj["rebuttal_sentences"], index_generator)


def write_examples_to_file(examples, filename):
  with open(filename, 'w') as f:
    for example in examples:
      f.write(json.dumps(example._asdict()) + "\n")


Example = collections.namedtuple(
    "Example",
    "overall_index identifier review_sentence rebuttal_sentence both_sentences score label"
    .split())


def make_vocabber(tokens, index_generator, output_dir):
  examples = []
  tokens = sorted(tokens)
  for i in range(0, len(tokens), 20):
    sentence = " ".join(tokens[i:i + 20])
    examples.append(
        Example(next(index_generator), None, sentence, sentence, sentence, 0,
                0))
  write_examples_to_file(examples, output_dir + "/vocabber.jsonl")


MAX_EXAMPLES_PER_FILE = 10000

def main():

  output_dir = "temp_data_files"
  overall_identifier_list = []
  index_generator = overall_indexifier()
  overall_token_vocab = set()
  
  os.makedirs(output_dir + "/")

  for subset in SUBSETS:
    example_lists = collections.defaultdict(list)
    print("Working on subset: ", subset)
    file_list = glob.glob("/".join([sys.argv[1], subset, "*"]))
    for i, input_filename in enumerate(tqdm.tqdm(file_list)):
      pair_ml_examples, identifiers, token_vocab = get_general_examples(
          input_filename, index_generator)
      overall_token_vocab.update(token_vocab)
      overall_identifier_list += identifiers
      example_lists[identifiers[0][0]] = pair_ml_examples

    current_list = []
    num_files_written = 0

    os.makedirs(output_dir + "/" + subset + "/")

    for review_id, examples in example_lists.items():
      if len(examples) + len(current_list) > MAX_EXAMPLES_PER_FILE:
        filename = make_output_filename(output_dir, subset, num_files_written)
        write_examples_to_file(current_list, filename)
        num_files_written += 1
        current_list = []
      else:
        current_list += examples
    if current_list:
      filename = make_output_filename(output_dir, subset, num_files_written)
      write_examples_to_file(current_list, filename)
    #write_examples_to_file(sum(example_lists.values(), []), output_dir +"/" +
    #subset + "_all.jsonl")


    all_examples = sum(example_lists.values(), [])
    print(collections.Counter(x.label for x in all_examples))

  make_vocabber(overall_token_vocab, index_generator, output_dir)


if __name__ == "__main__":
  main()
