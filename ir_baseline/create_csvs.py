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
FIELDS = "id label text".split()
RowTuple = collections.namedtuple('RowTuple', FIELDS)

PAIRS_PER_FILE = 20 # What actually is this?

class Model(object):
  rank = 'rank'
  rankprob = 'rankprob'
  ALL = [rank, rankprob]

def rank_process(offset, rebuttal_sentence, review_sentences):
  examples =  []
  for i, (review_sentence, score) in enumerate(review_sentences):
    examples.append(
        RowTuple(offset + i, 
        score, 
        ' [SEP] '.join(
            [rebuttal_sentence, review_sentence]).replace('\0', ""))._asdict())
  return examples


def rankprob_process(offset, rebuttal_sentence, review_sentences):
  examples =  []
  split = int(SAMPLES_PER_SENTENCE[Model.rankprob]/2)
  d1s = review_sentences[:split]
  d2s = review_sentences[split:]
  index = offset

  for d1, score_1 in d1s:
    for d2, score_2 in d2s:
      if score_1 > score_2:
        label = POS
      else:
        label = NEG
      examples.append(
      RowTuple(
        index, label,
        ' [SEP] '.join([rebuttal_sentence, d1, d2]).replace('\0', "")
      )._asdict())
      index += 1
  return examples


FUNC_MAP = {
  Model.rank: rank_process,
  Model.rankprob : rankprob_process
}

SAMPLES_PER_SENTENCE = {
  Model.rank: 10,
  Model.rankprob: 6
}

def get_samples(filename, process_fn, samples_per_sentence, offset):
  with open(filename, 'r') as f:
    obj = json.load(f)

  scores = np.array(obj["scores"])
  review_len = len(obj["review"])

  examples = []

  for reb_i, reb_sentence in enumerate(obj["rebuttal"]):
    rev_indices = random.choices(range(review_len),
        k=samples_per_sentence)
    rev_sentences = [
      (obj["review"][idx], scores[reb_i][idx]) for idx in rev_indices
    ]
    examples += process_fn(offset + len(examples), reb_sentence, rev_sentences)

  return examples

TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')

def build_overall_vocab(sentences):
  vocab = collections.Counter()
  for sent in sentences:
    tokens = TOKENIZER.tokenize(sent)
    vocab.update(tokens)

  fake_sentences = []
  vocab = list(sorted(vocab.keys()))
  for start_index in range(0, len(vocab), 100):
    fake_sentences.append(" ".join(vocab[start_index:start_index +
      100]).replace('\0', ""))
  return fake_sentences


def write_csv(filename, dict_rows, fields):
  with open(filename, 'w') as f:
    writer = csv.DictWriter(f, fields)
    writer.writeheader()
    for row in dict_rows:
      writer.writerow(row)

def main():

  args = parser.parse_args()
  assert args.model in Model.ALL


  all_sentences = []

  for subset in "train dev".split():

    subset_collector = []

    all_filenames = list(sorted(glob.glob("/".join(
        [args.datadir , "traindev_"+ subset + "*.json"]))
    ))

    for batch_i, input_file_start_index in enumerate(
      tqdm.tqdm(range(0, len(all_filenames), PAIRS_PER_FILE))):

      this_file_filenames = all_filenames[
          input_file_start_index:input_file_start_index + PAIRS_PER_FILE]

      sample_collector = []
      for filename in this_file_filenames:

        with open(filename, 'r') as f:
          k = json.load(f)
          all_sentences += k["review"]
          all_sentences += k["rebuttal"]
        
        sample_collector += get_samples(filename, FUNC_MAP[args.model],
        SAMPLES_PER_SENTENCE[args.model], len(sample_collector))

      output_file = "".join([
          args.datadir, subset, "_", args.model, "_batch_", str(batch_i),  ".csv"])
      write_csv(output_file, sample_collector, FIELDS)
      subset_collector += sample_collector
    subset_file = args.datadir + args.model + "_" + subset + "_all.csv"
    write_csv(subset_file, subset_collector[:1000], FIELDS)


  labels = {
    Model.rank: [0.0, 0.5],
    Model.rankprob: [POS, NEG]
  }

  fake_sentences = build_overall_vocab(all_sentences)
  for model in Model.ALL:
    dummy_filename = args.datadir + "/" + model + "_overall_dummy_vocabber.csv"
    fake_examples = [
    {"id": i, "text": sentence, "label":labels[model][i%2]} for i, sentence in
    enumerate(fake_sentences)
    ]
    write_csv(dummy_filename, fake_examples, FIELDS)


if __name__ == "__main__":
  main()
