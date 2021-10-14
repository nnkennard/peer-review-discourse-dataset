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

random.seed(43)

parser = argparse.ArgumentParser(description='prepare jsonls for torchtext')
parser.add_argument('-i',
                    '--input_dir',
                    default="../data_prep/final_dataset/",
                    type=str,
                    help='path to dataset files')
parser.add_argument('-o',
                    '--output_dir',
                    default="torchtext_input_data_posneg_1_sample_1.0",
                    type=str,
                    help='path to dataset files')

MAX_EXAMPLES_PER_FILE = 10000
NEG_TO_POS_SAMPLE_RATIO = 1
#POS_SAMPLE_RATIO = 1.0

STEMMER = PorterStemmer()
STOPWORDS = stopwords.words('english')


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


TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')


def get_token_vocab(review_sentences, rebuttal_sentences):
  tokens = set()
  for sentence in review_sentences + rebuttal_sentences:
    normal_tokens = word_tokenize(sentence.lower())
    bert_tokens = TOKENIZER.tokenize(sentence)
    tokens.update(
        set(normal_tokens).union(tok for tok in bert_tokens if '#' not in tok))
  return tokens


def make_pair_examples(
    review_id,
    review_sentences,
    rebuttal_sentences,
):
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
      if identifier in identifiers:
        dsdsds
      identifiers.append(identifier)
      label = 1 if RelatedPair(review_index,
                               rebuttal_index) in true_related_pairs else 0
      review_sentence = review_sentence_texts[review_index]
      rebuttal_sentence = rebuttal_sentence_texts[rebuttal_index]
      both_sentences = review_sentence + " [SEP] " + rebuttal_sentence
      example_maps[label].append(
          Example(None, identifier, review_sentence_texts[review_index],
                  both_sentences, rebuttal_sentence_texts[rebuttal_index],
                  score, label))
    pos_examples = example_maps[1]
    sampled_pos_examples = pos_examples
    #for pos_example in pos_examples:
    #  choice = random.random()
    #  if choice < POS_SAMPLE_RATIO:
    #    sampled_pos_examples.append(pos_example)

    sampled_neg_examples = random.sample(
        example_maps[0],
        max(
            min(len(example_maps[0]),
                NEG_TO_POS_SAMPLE_RATIO * len(sampled_pos_examples)), 3))

    examples += sampled_pos_examples + sampled_neg_examples
    filtered_examples = {}
    for example in examples:
      if example.identifier in filtered_examples:
        continue
      else:
        filtered_examples[example.identifier] = example

  examples = sorted(filtered_examples.values())
  random.shuffle(examples)

  return examples, review_id, get_token_vocab(review_sentence_texts,
                                              rebuttal_sentence_texts)


def make_output_filename(output_dir, subset, index):
  return "/".join([output_dir, subset, str(index).zfill(4) + ".jsonl"])


def get_general_examples(input_filename):
  with open(input_filename, 'r') as f:
    obj = json.load(f)
    return make_pair_examples(obj["metadata"]["review_id"],
                              obj["review_sentences"],
                              obj["rebuttal_sentences"])


def write_examples_to_file(examples, filename, index_offset):
  identifier_list = []
  with open(filename, 'w') as f:
    for i, example in enumerate(examples):
      index = index_offset + i
      temp_dict = example._asdict()
      temp_dict["overall_index"] = index
      identifier_list.append((index, temp_dict["identifier"]))
      del temp_dict["identifier"]
      f.write(json.dumps(temp_dict) + "\n")
  return identifier_list, index_offset + len(examples)


Example = collections.namedtuple(
    "Example",
    "overall_index identifier review_sentence rebuttal_sentence both_sentences score label"
    .split())


def make_vocabber(tokens, output_dir):
  examples = []
  tokens = sorted(tokens)
  j = 0
  for i in range(0, len(tokens), 20):
    sentence = " ".join(tokens[i:i + 20])
    examples.append(Example(j, None, sentence, sentence, sentence, 0, 0))
    j += 1
  write_examples_to_file(examples, output_dir + "/vocabber.jsonl", 0)


def create_if_not_exists(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)


def make_metadata(output_dir, overall_identifier_list):
  index_to_review_map = {}
  review_to_map_map = collections.defaultdict(dict)

  print("Overall identifier list")
  print(len(overall_identifier_list))

  seen_indices = []
  for index, identifier in overall_identifier_list:
    review_id, review_index, rebuttal_index = identifier
    if index in index_to_review_map:
      dsdsdsds
    index_to_review_map[index] = review_id
    key = "{0}_{1}".format(review_index, rebuttal_index)
    if key in review_to_map_map[review_id]:
      dsds
    review_to_map_map[review_id][key] = index

  print("Writing metadata")
  print("indices mapped to reviews")
  print(len(index_to_review_map))
  print("indices accounted for")
  print(len(sum([list(k.values()) for k in review_to_map_map.values()], [])))

  with open(output_dir + "/metadata.json", 'w') as f:
    json.dump(
        {
            "index_to_review_map": index_to_review_map,
            "review_to_map_map": dict(review_to_map_map),
            "negative_to_positive_example_ratio": NEG_TO_POS_SAMPLE_RATIO
        }, f)


def main():

  args = parser.parse_args()

  create_if_not_exists(args.output_dir)

  overall_token_vocab = set()
  examples_by_subset = {}

  for subset in SUBSETS:
    example_lists = collections.defaultdict(list)
    file_list = glob.glob("/".join([args.input_dir, subset, "*"]))
    for i, input_filename in enumerate(tqdm.tqdm(file_list)):
      pair_ml_examples, review_id, token_vocab = get_general_examples(
          input_filename)
      overall_token_vocab.update(token_vocab)
      example_lists[review_id] = pair_ml_examples

    examples_by_subset[subset] = example_lists

  index_offset = 0
  overall_identifier_list = []
  for subset, example_lists in examples_by_subset.items():
    create_if_not_exists(args.output_dir + "/" + subset + "/")

    num_files_written = 0
    current_list = []
    for review_id, examples in example_lists.items():
      if len(examples) + len(current_list) > MAX_EXAMPLES_PER_FILE:
        filename = make_output_filename(args.output_dir, subset,
                                        num_files_written)
        identifiers, index_offset = write_examples_to_file(
            current_list, filename, index_offset)
        overall_identifier_list += identifiers
        num_files_written += 1
        current_list = examples
      else:
        current_list += examples
    if current_list:
      filename = make_output_filename(args.output_dir, subset,
                                      num_files_written)
      identifiers, index_offset = write_examples_to_file(
          current_list, filename, index_offset)
      overall_identifier_list += identifiers

    print(len(overall_identifier_list))

  make_vocabber(overall_token_vocab, args.output_dir)

  print("Total examples", len(overall_identifier_list))

  make_metadata(args.output_dir, overall_identifier_list)


if __name__ == "__main__":
  main()
