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


def predict_sentiment(model_name, model, tokenizer, metadata, sentence, device):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:metadata.max_input_length-2]
    indexed = [metadata.init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [metadata.eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    if model_name == classification_lib.Model.rankprob:
      return torch.argmax(torch.sigmoid(model(tensor))).item()
    else:
     return model(tensor).item()

def get_bert_ranks(scores):
  sentence_scores = [sum(scores[i].values()) for i in range(len(scores))]
  scores_in_order = sorted(sentence_scores)
  score_to_rank = {}
  for score in scores_in_order:
    for i, x in enumerate(scores_in_order):
      if x == score:
        score_to_rank[score] = i + 1
        break
  return [score_to_rank[score] for score in sentence_scores]


STEMMER = PorterStemmer()
STOPWORDS = stopwords.words('english')


def preprocess(sentence):
  return [
      STEMMER.stem(word).lower() for word in sentence.split()
      if word.lower() not in STOPWORDS
  ]

def mean(l):
  if not l:
    return None
  else:
    return sum(l) / len(l)


Result = collections.namedtuple("Result", ("review_id rebuttal_idx "
    "actual_alignment rank_mrr rankprob_mrr bm25_mrr").split())


def get_rank_scores(model_name, model, tokenizer, metadata, query,
    device):
    bert_scores = {}
    for i, doc in enumerate(review_sentences):
      example = " [SEP] ".join([doc, query])
      bert_scores[i] = predict_sentiment(
        model_name, model, tokenizer, metadata, example, device)
    return bert_scores


def get_rankprob_scores(model_name, model, tokenizer, metadata, query,
    device):
    bert_scores = collections.defaultdict(dict)
    for i, d1 in enumerate(review_sentences):
      for j, d2 in enumerate(review_sentences):
        example = " [SEP] ".join([d1, d2, query])
        bert_scores[i][j] = predict_sentiment(
          model_name, model, tokenizer, metadata, example, device)

    return bert_scores


def main():

  args = parser.parse_args()

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  metadata = classification_lib.TokenizerMetadata(tokenizer)

  models = {}
  for model_name in classification_lib.Model.ALL:
    bertmodel = classification_lib.BERTGRUClassifier(
        device, model_name).to(device)
    bertmodel.load_state_dict(
      torch.load(classification_lib.get_checkpoint_name(model_name)))
    models[model_name] = bertmodel

  results = []
  for filename in tqdm(glob.glob("0517_split_2/test/*")): 
    with open(filename, 'r') as f:
      obj = json.load(f)
    review_sentences = [sent["sentence"] for sent in obj["review"]]
    if len(review_sentences) > 20:
      continue
    rebuttal_sentences = [sent["sentence"] for sent in obj["rebuttal"]]
    alignment_labels = [x["labels"]["alignments"]
                        for x in obj["rebuttallabels"]]


    # BM 25 ranking
    tokenized_corpus = [preprocess(sent) for sent in review_sentences]
    model = BM25Okapi(tokenized_corpus)
    for reb_i, (query, labels) in enumerate(
          zip(rebuttal_sentences, alignment_labels)):
      if not labels:
        continue
      else:
        preprocessed_query = preprocess(query)
        ranked = model.get_top_n(preprocessed_query, tokenized_corpus,
                  len(review_sentences))
        bert_scores = get_bert_scores(model_name,models[model_name], tokenizer,
        metadata, example, device)
        bm25_ranks = []
        for tokenized_sent in tokenized_corpus:
          bm25_ranks.append(ranked.index(tokenized_sent) + 1)
          top_sentences = get_bert_ranks(bert_scores)

      rrs = {"bm25":[], "ws":[]}
      if labels:
        for i in labels:
          rrs["bm25"].append(1/bm25_ranks[i])
          rrs["ws"].append(1/top_sentences[i])

      results.append(Result(
        obj["metadata"]["review"],
        reb_i,
        "|".join(str(i) for i in labels),
        mean(rrs["ws"]),
        mean(rrs["bm25"]),
      )._asdict())
      print(results[-1])


  pd.DataFrame.from_dict(results).to_csv("mrr_results_small.csv")


if __name__ == "__main__":
  main()
