import glob
import json
import os
from tqdm import tqdm
import numpy as np


def alignments_correct(pair):
  rev_len = len(pair["review_sentences"])
  for reb_sent in pair["rebuttal_sentences"]:
    _, indices = reb_sent["alignment"]
    if indices is not None:
      if max(indices) >= rev_len:
        return False
  return True


NO_MATCH = "NO_MATCH"
NUM_NEG_SAMPLES = 2


def read_data(data_dir):
  sentence_pairs = []
  print("Loading data from " + data_dir)
  for filename in tqdm(list(glob.glob(data_dir + "/*.json"))):
    with open(filename, 'r') as f:
      pair = json.load(f)
      if not alignments_correct(pair):
        continue
      review_texts = [x["text"] for x in pair["review_sentences"]]
      for reb_i, rebuttal_sent in enumerate(pair["rebuttal_sentences"]):
        align_type, align_indices = rebuttal_sent["alignment"]
        if align_indices is None:
          sentence_pairs.append((rebuttal_sent["text"], NO_MATCH, 1))
          neg_candidate_indices = list(range(len(pair["review_sentences"])))
        else:
          for align_idx in align_indices:
            sentence_pairs.append(
                (rebuttal_sent["text"], review_texts[align_idx], 1))
          neg_candidate_indices = list(
              sorted(set(range(len(review_texts))) - set(align_indices)))

        for _ in range(min(len(neg_candidate_indices), NUM_NEG_SAMPLES)):
          negative_ind = np.random.randint(0, len(review_texts))
          sentence_pairs.append(
              (rebuttal_sent["text"], review_texts[negative_ind], 0))
  return sentence_pairs
