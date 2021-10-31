import argparse
import collections
import glob
import json
import os
import rank_bm25
import tqdm

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#random.seed(43)

parser = argparse.ArgumentParser(description='prepare jsonls for torchtext')
parser.add_argument('-i',
                    '--input_dir',
                    default="../data_prep/final_dataset/",
                    type=str,
                    help='path to dataset files')

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

def mean(l):
  return sum(l)/len(list(l))

def mrr(ranks):
  return mean([1/x for x in ranks])

def check_alignment_indices(doc_obj):
  num_review_sentences = len(doc_obj["review_sentences"])
  for sent in doc_obj["rebuttal_sentences"]:
    _, indices = sent["alignment"]
    if indices is not None:
      if max(indices) >= num_review_sentences:
        return False
  return True

def get_true_pairs(rebuttal_sentences):
  true_pairs = {}
  for i, sent in enumerate(rebuttal_sentences):
    _, indices = sent["alignment"]
    if indices is not None and indices:
      true_pairs[i] = indices
  return true_pairs


def get_query_rrs(scores, true_corpus_indices):
  query_rrs = []
  sorted_scores = sorted(
    list(enumerate(scores)), key=lambda x:x[1], reverse=True)
  current_rank = 0
  current_score = float("inf")
  successes_at_rank = collections.Counter()
  for i, (corpus_index, score) in enumerate(sorted_scores):
    if score < current_score:
      current_rank = i + 1
    if corpus_index in true_corpus_indices:
      successes = sum(v for k, v in successes_at_rank.items() if k <
      current_rank)
      query_rrs.append(1/(current_rank - successes))
      successes_at_rank[current_rank] += 1
    current_score = score
  return query_rrs

def get_rrs_from_document(doc_obj, true_pairs, threshold):
  rebuttal_sentences = [s["text"] for s in doc_obj["rebuttal_sentences"]]

  queries = [preprocess(sent) for sent in rebuttal_sentences]
  model = create_document_model(doc_obj)

  doc_rrs = []
  for query_index, true_corpus_indices in true_pairs.items():
    scores = model.get_scores(queries[query_index])
    doc_rrs.append(get_query_rrs(scores, true_corpus_indices))
  return doc_rrs


def get_rrs_from_corpus(doc_obj, true_pairs, threshold, full_corpus_model):
  rebuttal_sentences = [s["text"] for s in doc_obj["rebuttal_sentences"]]
  queries = [preprocess(sent) for sent in rebuttal_sentences]

  start_index, end_index = full_corpus_model.index_map[
                            doc_obj["metadata"]["review_id"]]

  doc_rrs = []
  for query_index, true_corpus_indices in true_pairs.items():
    scores = full_corpus_model.model.get_scores(queries[query_index])
    relevant_scores = scores[start_index:end_index]
    doc_rrs.append(get_query_rrs(relevant_scores, true_corpus_indices))
  return doc_rrs


def print_rr_means(rr_collector):
  corpus_rrs = []
  doc_mrrs = []
  query_mrrs = []
  for doc_rrs in rr_collector:
    for query_rrs in doc_rrs:
      query_mrrs.append(mean(query_rrs))
      corpus_rrs += query_rrs
    doc_mrrs.append(mean(sum(doc_rrs, [])))

  return {
    "corpus": mean(corpus_rrs),
    "doc": mean(doc_mrrs),
    "query": mean(query_mrrs)
  }

def get_subset_mrr(glob_path, corpus_level, full_corpus_model):
  rr_collector = []
  for filename in glob.glob(glob_path):
    with open(filename, 'r') as f:
      obj = json.load(f)
      if not check_alignment_indices(obj):
        continue
      true_pairs = get_true_pairs(obj["rebuttal_sentences"])
    if corpus_level == "corpus":
      rrs = get_rrs_from_corpus(obj, true_pairs, 0.0, full_corpus_model)
    else:
      rrs = get_rrs_from_document(obj, true_pairs, 0.0)
    if rrs:
      rr_collector.append(rrs)

  return print_rr_means(rr_collector)

class FullCorpusModel(object):
  def __init__(self, model, index_map):
    self.model = model
    self.index_map = index_map

def get_full_corpus_model(input_dir):
  corpus = []
  index_map = {}
  for subset in SUBSETS:
    for filename in glob.glob(input_dir + "/" + subset + "/*.json"):
      with open(filename, 'r') as f:
        obj = json.load(f)
      review_id = obj["metadata"]["review_id"]
      start_index = len(corpus)
      new_sentences = [preprocess(x["text"]) for x in obj["review_sentences"]]
      end_index = start_index + len(new_sentences)
      corpus += new_sentences
      index_map[review_id] = (start_index, end_index)
  return FullCorpusModel(rank_bm25.BM25Okapi(corpus), index_map)


def create_document_model(obj):
  return rank_bm25.BM25Okapi([preprocess(x["text"]) for x in
  obj["review_sentences"]])

def get_threshold_info(glob_path, corpus_level, full_corpus_model):
  scores_of_negatives = []
  scores_of_positives = []
  for filename in glob.glob(glob_path):
    with open(filename, 'r') as f:
      obj = json.load(f)


    if corpus_level == "corpus":
      model = full_corpus_model.model
      start_index, end_index = full_corpus_model.index_map[
        obj["metadata"]["review_id"]]
    else:
      model = create_document_model(obj)

    for query_index, reb_sent in enumerate(obj["rebuttal_sentences"]):
      query = preprocess(reb_sent["text"])
      align_type, indices = reb_sent["alignment"]
      if corpus_level == "corpus":
        scores = full_corpus_model.model.get_scores(query)
        scores_of_negatives += list(scores[:start_index]) + list(
          scores[end_index:])
        relevant_scores = scores[start_index:end_index]
      else:
        relevant_scores = model.get_scores(query)

      if indices is None:
        scores_of_negatives += list(relevant_scores)
      else:
        for i, relevant_score in enumerate(relevant_scores):
          if i in indices:
            scores_of_positives.append(relevant_score)
          else:
            scores_of_negatives.append(relevant_score)

  return {
  "negatives": make_counter_from_scores(scores_of_negatives),
  "positives": make_counter_from_scores(scores_of_positives),
  }


def make_counter_from_scores(scores):
  return collections.Counter(
    ["{:.3f}".format(k) for k in scores]
    )

def main():

  args = parser.parse_args()

  rank_collector = []

  full_corpus_model = get_full_corpus_model(args.input_dir)

  thresholds = {}
  for corpus_level in "corpus doc".split():
    thresholds[corpus_level] = get_threshold_info(args.input_dir + "/train/*",
    corpus_level, full_corpus_model)

  with open("thresholds.json", 'w') as f:
    json.dump(thresholds, f)

  for subset in SUBSETS:
    for corpus_level in "corpus doc".split():
      print(corpus_level)
      k = get_subset_mrr("/".join([args.input_dir, subset, "*"]), corpus_level,
      full_corpus_model)
      print(k)
    break

if __name__ == "__main__":
  main()
