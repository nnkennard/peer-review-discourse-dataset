import argparse
import collections
import glob
import json
import os
#import random
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


def get_ranks_of_positives(example_maps):
  all_examples = sum(example_maps.values(), [])
  ranks = collections.defaultdict(list)
  for pos_example in example_maps[1]:
    review_id, rev_idx, reb_idx = pos_example.identifier
    relevant_examples = []
    for i in all_examples:
      a, b, c = i.identifier
      assert a == review_id
      if c == reb_idx:
        relevant_examples.append(i)
    sorted_examples = sorted(relevant_examples, key=lambda x:x.score,
    reverse=True)

    for i, e in enumerate(sorted_examples):
      if e == pos_example:
        ranks[e.identifier[2]].append(i+1)
  return ranks


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
                  rebuttal_sentence_texts[rebuttal_index], both_sentences,
                  score, label))
    pos_examples = example_maps[1]
    sampled_pos_examples = pos_examples
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

  ranks = get_ranks_of_positives(example_maps)

  return examples, review_id, get_token_vocab(review_sentence_texts,
                                    rebuttal_sentence_texts), ranks


def make_output_filename(output_dir, subset, index):
  return "/".join([output_dir, subset, str(index).zfill(4) + ".jsonl"])


def get_general_examples(input_filename):
  with open(input_filename, 'r') as f:
    obj = json.load(f)
    return make_pair_examples(obj["metadata"]["review_id"],
                              obj["review_sentences"],
                              obj["rebuttal_sentences"])


Example = collections.namedtuple(
    "Example",
    "overall_index identifier review_sentence rebuttal_sentence both_sentences score label"
    .split())


def create_if_not_exists(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)


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


def get_rrs_from_document(doc_obj, threshold):
  if not check_alignment_indices(doc_obj):
    return []
  review_sentences = [s["text"] for s in doc_obj["review_sentences"]]
  rebuttal_sentences = [s["text"] for s in doc_obj["rebuttal_sentences"]]
  true_pairs = get_true_pairs(doc_obj["rebuttal_sentences"])

  queries = [preprocess(sent) for sent in rebuttal_sentences]
  corpus = [preprocess(sent) for sent in review_sentences]
  model = rank_bm25.BM25Okapi(corpus)

  doc_rrs = []
  for query_index, true_corpus_indices in true_pairs.items():
    query_rrs = []
    scores = model.get_scores(queries[query_index])
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
    doc_rrs.append(query_rrs)
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

def get_subset_mrr(glob_path):
  rr_collector = []
  for filename in glob.glob(glob_path):
    with open(filename, 'r') as f:
      obj = json.load(f)
    rrs = get_rrs_from_document(obj, threshold=0.0)
    if rrs:
      rr_collector.append(rrs)

  k = print_rr_means(rr_collector)
  print(k)

def main():

  args = parser.parse_args()

  rank_collector = []

  for subset in SUBSETS:
    get_subset_mrr("/".join([args.input_dir, subset, "*"]))
    break

if __name__ == "__main__":
  main()
