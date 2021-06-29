import argparse
import collections
import csv
import glob
import json
import numpy as np
import random
import sys
import tqdm

from transformers import BertTokenizer

parser = argparse.ArgumentParser(description='prepare CSVs for ws training')
parser.add_argument('-d',
                    '--datadir',
                    default="../0517_split_2/train/",
                    type=str,
                    help='path to data file containing score jsons')
parser.add_argument('-m',
                    '--model',
                    default="rankprob",
                    type=str,
                    help='which IR model to prepare for')

POS = "1_more_relevant"
NEG = "2_more_relevant"

PAIRS_PER_FILE = 20  # What actually is this?


class Model(object):
  rank = 'rank'
  rankprob = 'rankprob'
  ALL = [rank, rankprob]


NUM_SAMPLES = {
    Model.rank: 10,
    Model.rankprob: 5,
}

TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_text(lines):
  return [TOKENIZER.tokenize(line) for line in lines]


def get_indices(text, num_samples):
  return [(i, text[i]) for i in random.choices(range(len(text)), num_samples)]


LIL_SEP_LIST = ["[SEP]"]


def create_rank_text(reb_sentence, rev_sentence):
  return sum([reb_sentence, LIL_SEP_LIST, rev_sentence], [])


def create_rankprob_text(reb_sentence, rev_sentence_1, rev_sentence_2):
  return sum([
      reb_sentence, LIL_SEP_LIST, rev_sentence_1, LIL_SEP_LIST, rev_sentence_2
  ], [])


def rankprob_sampler(tokenized_review, tokenized_rebuttal, scores):
  examples = []
  for reb_index, reb_sentence in enumerate(tokenized_rebuttal):
    rev_mapper = list(zip(tokenized_review, scores[reb_index]))
    for _ in range(NUM_SAMPLES[Model.rankprob]):
      rev_i, rev_j = random.choices(range(len(tokenized_review)), k=2)
      rev_sent_i, rev_score_i = rev_mapper[rev_i]
      rev_sent_j, rev_score_j = rev_mapper[rev_j]
      if rev_score_i > rev_score_j:
        rev_sent_high = rev_sent_i
        rev_sent_low = rev_sent_j
      else:
        rev_sent_high = rev_sent_j
        rev_sent_low = rev_sent_i
      examples += [{
          "text":
              create_rankprob_text(reb_sentence, rev_sent_high, rev_sent_low),
          "label":
              POS
      }, {
          "text":
              create_rankprob_text(reb_sentence, rev_sent_low, rev_sent_high),
          "label":
              NEG
      }]
  return examples


def rank_sampler(tokenized_review, tokenized_rebuttal, scores):
  examples = []
  for reb_index, reb_sentence in enumerate(tokenized_rebuttal):
    rev_mapper = list(zip(tokenized_review, scores[reb_index]))
    for _ in range(NUM_SAMPLES[Model.rank]):
      rev_i = random.choice(range(len(tokenized_review)))
      sent, score = rev_mapper[rev_i]
      examples.append({
          "text": create_rank_text(reb_sentence, sent),
          "label": score
      })
  return examples


SAMPLER_MAP = {
    Model.rank: rank_sampler,
    Model.rankprob: rankprob_sampler,
}


def build_batch_file(filenames, offsets, batch_filename_template, vocab):
  batch_builders = {model: [] for model in Model.ALL}
  for filename in filenames:
    with open(filename, 'r') as f:
      obj = json.load(f)
      tokenized_review = tokenize_text(obj["review"])
      tokenized_rebuttal = tokenize_text(obj["rebuttal"])
      for model in Model.ALL:
        batch_builders[model] += SAMPLER_MAP[model](tokenized_review,
                                                    tokenized_rebuttal,
                                                    obj["scores"])

      vocab.update(sum(tokenized_review + tokenized_rebuttal, []))

  for model, batch_builder in batch_builders.items():
    jsons = []
    output_filename = batch_filename_template.format(model)
    for i, example in enumerate(batch_builder):
      new_dict = dict(example)
      new_dict["index"] = offsets[model] + i
      jsons.append(json.dumps(new_dict))

    offsets[model] += len(jsons)
    with open(output_filename, 'w') as f:
      f.write("\n".join(jsons))

  return offsets


def build_fake_sentences(vocab):
  toks = list(vocab.keys())
  fake_sentences = []
  for start_index in range(0, len(toks), 100):
    fake_sentences.append(" ".join(toks[start_index:start_index + 100]).replace(
        '\0', ""))
  return fake_sentences


def main():

  args = parser.parse_args()
  assert args.model in Model.ALL

  all_sentences = []
  vocab = collections.Counter()

  for subset in "train dev".split():

    all_filenames = list(
        sorted(
            glob.glob("/".join([args.datadir,
                                "traindev_" + subset + "*.json"]))))

    offsets = {model: 0 for model in Model.ALL}

    for batch_i, input_file_start_index in enumerate(
        tqdm.tqdm(range(0, len(all_filenames), PAIRS_PER_FILE))):

      this_file_filenames = all_filenames[
          input_file_start_index:input_file_start_index + PAIRS_PER_FILE]
      output_file = "".join(
          [args.datadir, subset, "_{0}_batch_",
           str(batch_i), ".jsonl"])
      offset = build_batch_file(this_file_filenames, offsets, output_file,
                                vocab)

  fake_labels = {Model.rank: [0.0, 0.5], Model.rankprob: [POS, NEG]}
  fake_sentences = build_fake_sentences(vocab)
  for model in Model.ALL:
    output_file = "".join([args.datadir, model + "_vocabber.jsonl"])
    fake_examples = [{
        "index": i,
        "text": sentence,
        "label": fake_labels[model][i % 2]
    } for i, sentence in enumerate(fake_sentences)]
    with open(output_file, 'w') as f:
      f.write("\n".join(json.dumps(ex) for ex in fake_examples))


if __name__ == "__main__":
  main()
