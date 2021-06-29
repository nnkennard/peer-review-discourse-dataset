import glob
import pandas as pd
import argparse
import collections
import json
import torch
import torch.optim as optim
from torchtext.legacy import data
from transformers import BertTokenizer
from transformers import BertModel
import torch.nn as nn
from tqdm import tqdm

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from rank_bm25 import BM25Okapi
import classification_lib

parser = argparse.ArgumentParser(description='prepare CSVs for ws training')
parser.add_argument('-d',
                    '--datadir',
                    default="data/ws/",
                    type=str,
                    help='path to data file containing score jsons')

STEMMER = PorterStemmer()
STOPWORDS = stopwords.words('english')


def preprocess(sentence):
  return [
      STEMMER.stem(word).lower()
      for word in sentence.split()
      if word.lower() not in STOPWORDS
  ]


def mean(l):
  if not l:
    return None
  else:
    return sum(l) / len(l)


Result = collections.namedtuple(
    "Result", ("review_id rebuttal_idx "
               "actual_alignment rank_mrr rankprob_mrr bm25_mrr").split())


def get_prediction(model, sentence, dataset_tools, prediction_getter):

  tokenizer = dataset_tools.tokenizer
  model.eval()
  tokens = tokenizer.tokenize(sentence)
  tokens = tokens[:dataset_tools.metadata.max_input_length - 2]
  indexed = [dataset_tools.metadata.init_token_idx
            ] + tokenizer.convert_tokens_to_ids(tokens) + [
                dataset_tools.metadata.eos_token_idx
            ]
  tensor = torch.LongTensor(indexed).to(dataset_tools.device)
  tensor = tensor.unsqueeze(0)
  return prediction_getter(model(tensor))


# === Getting ranks ===


def get_bm25_ranks(review_sentences, rebuttal_sentences):
  tokenized_corpus = [preprocess(sent) for sent in review_sentences]
  model = BM25Okapi(tokenized_corpus)
  rebuttal_ranks = []
  for reb_i, query in enumerate(rebuttal_sentences):
    preprocessed_query = preprocess(query)
    ranked = model.get_top_n(preprocessed_query, tokenized_corpus,
                             len(review_sentences))
    bm25_ranks = []
    for tokenized_sent in tokenized_corpus:
      bm25_ranks.append(ranked.index(tokenized_sent) + 1)
    rebuttal_ranks.append(bm25_ranks)

  assert len(rebuttal_ranks) == len(rebuttal_sentences)
  assert all(len(i) == len(review_sentences) for i in rebuttal_ranks)
  return rebuttal_ranks


def convert_scores_to_ranks(scores):
  re_sorted = list(
      sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True))
  ordered_indices = list(x[0] for x in re_sorted)
  assert sorted(ordered_indices) == list(range(len(scores)))
  ranks = list(ordered_indices.index(i) + 1 for i in range(len(scores)))
  print(ranks)
  print(len(scores))
  print("*")
  return ranks


def get_mrr_from_ranks(ranks, label_indices):
  print("Ranks", ranks)
  print(len(label_indices))
  print(label_indices)
  print()
  return mean([1 / ranks[i] for i in label_indices])


def convert_scores_to_mrrs(scores_list, alignment_labels):
  individual_mrr_list = []
  for scores, label_list in zip(scores_list, alignment_labels):
    print("S", scores)
    print("LL", label_list)
    if not label_list:
      continue
    individual_mrr_list.append(
          get_mrr_from_ranks(convert_scores_to_ranks(scores), label_list))
  return mean(individual_mrr_list)


def process_rank_model(review_sentences, rebuttal_sentences, alignment_labels,
                       model, dataset_tools):
  bert_scores = collections.defaultdict(list)
  prediction_getter = classification_lib.PREDICTION_GETTER[
      classification_lib.Model.rank]
  for j, query in enumerate(rebuttal_sentences):
    for i, doc in enumerate(review_sentences):
      example = " [SEP] ".join([doc, query])
      bert_scores[j].append(
          get_prediction(model, example, dataset_tools, prediction_getter))
  score_list = []
  for i in range(len(rebuttal_sentences)):
    score_list.append(bert_scores[i])
  return convert_scores_to_mrrs(score_list, alignment_labels)


def main():

  args = parser.parse_args()

  # Set up tools for model
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  metadata = classification_lib.TokenizerMetadata(tokenizer)
  dataset_tools = classification_lib.DatasetTools(tokenizer, device, metadata,
                                                  None)

  # Set up models
  models = {}
  for model_name in classification_lib.Model.ALL:
    bertmodel = classification_lib.BERTGRUClassifier(device,
                                                     model_name).to(device)
    bertmodel.load_state_dict(
        torch.load(classification_lib.get_checkpoint_name(model_name)))
    models[model_name] = bertmodel

  results = []
  for filename in tqdm(glob.glob("0517_split_2/test/*")):

    # Read in data and labels
    with open(filename, 'r') as f:
      obj = json.load(f)
    review_sentences = [sent["sentence"] for sent in obj["review"]]
    if len(review_sentences) > 20:
      continue
    rebuttal_sentences = [sent["sentence"] for sent in obj["rebuttal"]]
    alignment_labels = [
        x["labels"]["alignments"] for x in obj["rebuttallabels"]
    ]

    assert len(rebuttal_sentences) == len(alignment_labels)

    bm25_ranks = get_bm25_ranks(review_sentences, rebuttal_sentences)
    rank_model_ranks = process_rank_model(review_sentences, rebuttal_sentences,
                                          alignment_labels,
                                          models[classification_lib.Model.rank],
                                          dataset_tools)

  pd.DataFrame.from_dict(results).to_csv("mrr_results_small.csv")


if __name__ == "__main__":
  main()
