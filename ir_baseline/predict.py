import glob
import pandas as pd
import argparse
import collections
import json
import torch
from transformers import BertTokenizer
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


def convert_scores_to_ranks(scores):
  re_sorted = list(
      sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True))
  ordered_indices = list(x[0] for x in re_sorted)
  assert sorted(ordered_indices) == list(range(len(scores)))
  ranks = list(ordered_indices.index(i) + 1 for i in range(len(scores)))
  return ranks

test_scores = [5, 10, 1, 2.5]
assert convert_scores_to_ranks(test_scores) == [2, 1, 4, 3]


def mean(l):
  if not l:
    return None
  else:
    return sum(l) / len(l)


def get_mrr_from_ranks(ranks, label_indices):
  return mean([1 / ranks[i] for i in label_indices])


def convert_rank_lists_to_mrr(review_sentences, rebuttal_sentences, rank_lists,
                              true_labels):
  assert len(rank_lists) == len(rebuttal_sentences)
  assert all(len(i) == len(review_sentences) for i in rank_lists)
  assert len(true_labels) == len(rebuttal_sentences)
  assert all(max(i) < len(review_sentences) for i in true_labels if i)
  assert all(max(i) <= len(review_sentences) for i in rank_lists if i)
  mrr_list = []
  error_list = []
  for i, (rank_list, label_list) in enumerate(zip(rank_lists, true_labels)):
    if not label_list:
      continue
    mrr = get_mrr_from_ranks(rank_list, label_list)
    mrr_list.append(mrr)
    error_list.append((i, mrr, rank_list[:len(label_list)], label_list))
  return error_list, mean(list(i for i in mrr_list if i is not None))


MRRResult = collections.namedtuple("MRRResult",
    "review_id model rebuttal_idx mrr top_ranked_indices actual_labels")

Result = collections.namedtuple("Result", ("review_id "
               "rank_mrr rankprob_mrr bm25_mrr").split())


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


def process_bm25_model(review_sentences, rebuttal_sentences, alignment_labels):
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

  return convert_rank_lists_to_mrr(review_sentences, rebuttal_sentences,
                                         rebuttal_ranks, alignment_labels)



def process_rank_model(review_sentences, rebuttal_sentences, alignment_labels,
                       model, dataset_tools):
  bert_scores = collections.defaultdict(list)
  prediction_getter = classification_lib.PREDICTION_GETTER[
      classification_lib.Model.rank]
  for j, query in enumerate(rebuttal_sentences):
    for i, doc in enumerate(review_sentences):
      example = " [SEP] ".join([query, doc])
      bert_scores[j].append(
          get_prediction(model, example, dataset_tools, prediction_getter))
  score_list = []
  for i in range(len(rebuttal_sentences)):
    score_list.append(bert_scores[i])
  rank_lists = list(convert_scores_to_ranks(i) for i in score_list)
  return convert_rank_lists_to_mrr(review_sentences, rebuttal_sentences,
  rank_lists, alignment_labels)

def convert_error_list(review_id, model, errors):
    return [MRRResult(review_id, model, *i)._asdict()  for i in errors]


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
  error_analysis_info = []
  for filename in tqdm(glob.glob("0517_split_2/dev/*")):

    # Read in data and labels
    with open(filename, 'r') as f:
      obj = json.load(f)

    review_id = obj["metadata"]["review"]

    review_sentences = [sent["sentence"] for sent in obj["review"]]
    if len(review_sentences) > 50:
      continue
    rebuttal_sentences = [sent["sentence"] for sent in obj["rebuttal"]]
    alignment_labels = [
        x["labels"]["alignments"] for x in obj["rebuttallabels"]
    ]

    #review_sentences = ["A sentence", "Another sentence",
    #"Yet another sentence, following the other two",
    #"A final sentence to round out the text"]
    #rebuttal_sentences = list(review_sentences)
    #review_id = "test_id"

    #alignment_labels = [[0], [1], [2], [3]]

    assert len(rebuttal_sentences) == len(alignment_labels)

    individual_mrrs, bm25_mrr = process_bm25_model(review_sentences, rebuttal_sentences,
    alignment_labels)
    error_analysis_info += convert_error_list(review_id, "bm25",  individual_mrrs)
    individual_mrrs, rank_model_mrr = process_rank_model(review_sentences, rebuttal_sentences,
                                        alignment_labels,
                                        models[classification_lib.Model.rank],
                                        dataset_tools)
    error_analysis_info += convert_error_list(review_id, "rank", individual_mrrs)
    results.append(Result(review_id, rank_model_mrr, None, bm25_mrr)._asdict())

    #break


  with open("mrr_errors_all.csv", 'w') as f:
    for err in error_analysis_info:
      f.write(json.dumps(err) + "\n")

  with open("mrr_results_all.csv", 'w') as f:
    for res in results:
      f.write(json.dumps(res) + "\n")




if __name__ == "__main__":
  main()
