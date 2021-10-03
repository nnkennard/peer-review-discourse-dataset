import argparse
import collections
import json
import random

from comet_ml import Experiment

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from contextlib import nullcontext
from torchtext.legacy import data
from transformers import BertTokenizer

import alignment_lib

parser = argparse.ArgumentParser(description='prepare CSVs for ws training')
parser.add_argument('-i',
                    '--input_dir',
                    default="score_model_input/",
                    type=str,
                    help='path to data file containing score jsons')
parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument('-m',
                    '--model_choice',
                    default="score_2t",
                    type=str,
                    help='which model to train')


def generate_text_field(tokenizer):
  metadata = alignment_lib.TokenizerMetadata(tokenizer)
  return data.Field(use_vocab=False,
                    batch_first=True,
                    tokenize=lambda x: alignment_lib.tokenize_and_cut(
                        tokenizer, metadata.max_input_length - 2, x),
                    preprocessing=tokenizer.convert_tokens_to_ids,
                    init_token=metadata.init_token_idx,
                    eos_token=metadata.eos_token_idx,
                    pad_token=metadata.pad_token_idx,
                    unk_token=metadata.unk_token_idx)


def make_fields(tokenizer):
  RAW = data.RawField()
  REVIEW_SENT = generate_text_field(tokenizer)
  REBUTTAL_SENT = generate_text_field(tokenizer)
  SCORE = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
  LABEL = data.Field(sequential=False, use_vocab=False, dtype=torch.int)

  return {
      "overall_index": ("overall_index", RAW),
      "review_sentence": ("review_sentence", REVIEW_SENT),
      "rebuttal_sentence": ("rebuttal_sentence", REBUTTAL_SENT),
      "label": ("label", LABEL),
      "score": ("score", SCORE),
  }


# Setting random seeds
SEED = 43
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def get_dataset_tools(data_dir):
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  metadata = alignment_lib.TokenizerMetadata(tokenizer)
  fields = make_fields(tokenizer)
  return alignment_lib.DatasetTools(tokenizer, device, metadata, fields)


def build_iterators(data_dir,
                    dataset_tools,
                    batch_size,
                    debug=False,
                    make_valid=False):

  if debug:
    train_file_name = "train_head.jsonl"
  else:
    train_file_name = "train/0000.jsonl"
  train_dataset, dev_dataset = data.TabularDataset.splits(
      path=data_dir,
      train=train_file_name,
      validation=train_file_name.replace("train", "dev"),
      format='json',
      skip_header=True,
      fields=dataset_tools.fields)

  for key in "review_sentence rebuttal_sentence label".split():
    dataset_tools.fields[key][1].build_vocab(train_dataset)

  return data.BucketIterator.splits([train_dataset, dev_dataset],
                                    batch_size=batch_size,
                                    device=dataset_tools.device,
                                    sort_key=lambda x: x.overall_index,
                                    sort_within_batch=False)

  #def get_match_map():
  #  with open("score_model_input/metadata.json", 'r') as f:
  #    metadata = json.load(f)

  return metadata["example_identifiers"], metadata["true_matches"]

  #example_identifiers = metadata["example_identifiers"]
  #true_match_map = collections.defaultdict(list)
  #for subset, matches in metadata["true_matches"].items():
  #  for review_id, rebuttal_index, review_index in matches:
  #    true_match_map[(review_id, rebuttal_index)].append(review_index)
  #
  #return example_identifiers, true_match_map


def get_index_to_rank_map(score_map):
  sorted_scores = sorted(score_map.values(), reverse=True)
  index_to_rank_map = {}
  for index, score in score_map.items():
    index_to_rank_map[int(index)] = sorted_scores.index(score) + 1
  return index_to_rank_map


def mean(l):
  if not l:
    return None
  return sum(l) / len(l)


def get_mrrs(epoch_data, example_identifiers, true_match_map):

  prediction_maps = collections.defaultdict(
      lambda: collections.defaultdict(list))
  for index, predicted_score in epoch_data.train_score_map.items():
    example_identifier = example_identifiers[index]
    review_id, rev_index, reb_index = example_identifier
    prediction_maps[(review_id, reb_index)][rev_index] = predicted_score

  mrr_accumulator = []
  for key, score_map in prediction_maps.items():
    index_to_rank_map = get_index_to_rank_map(score_map)
    if key[1] in true_match_map[key[0]]:
      rrs = []
      for k in true_match_map[key[0]][key[1]]:
        #if k in index_to_rank_map:
        rrs.append(1.0 / index_to_rank_map[k])
      print(rrs)
      mrr_accumulator.append(mean(rrs))

  print(mrr_accumulator)

  return mean(list(i for i in mrr_accumulator if i is not None))


def get_model_and_loss(model_choice, device):
  if model_choice == "score_2t":
    return alignment_lib.BERT2TRegresser(device).to(device), nn.MSELoss()
  else:
    assert False


def main():

  args = parser.parse_args()

  dataset_tools = get_dataset_tools(args.input_dir)

  model_save_name = alignment_lib.get_checkpoint_name()

  experiment = Experiment(project_name='ir_rank_model')

  all_train_iterator, all_valid_iterator, = build_iterators(
      args.input_dir,
      dataset_tools,
      alignment_lib.Hyperparams.batch_size,
      debug=args.debug,
      make_valid=True)

  model, loss = get_model_and_loss(args.model_choice, dataset_tools.device)

  optimizer = optim.Adam(model.parameters())

  best_valid_loss = float('inf')
  best_valid_epoch = None

  patience = 5000

  for epoch in range(10000):
    _ = alignment_lib.do_epoch(model, all_train_iterator, optimizer,
                               all_valid_iterator, loss)

    this_epoch_data = alignment_lib.do_epoch(model,
                                             all_train_iterator,
                                             optimizer,
                                             all_valid_iterator,
                                             loss,
                                             eval_both=True)
    alignment_lib.report_epoch(epoch, this_epoch_data)

    experiment.log_metric("Batch train MSE",
                          this_epoch_data.train_mse,
                          step=epoch)
    experiment.log_metric("Batch val MSE", this_epoch_data.val_mse, step=epoch)

    #train_mrr = get_mrrs(this_epoch_data, example_identifiers, true_match_map)
    #print(train_mrr)
    #experiment.log_metric("Train MRR",train_mrr, step=epoch)

    if this_epoch_data.val_mse < best_valid_loss:
      print("Best validation loss; saving model from epoch ", epoch)
      best_valid_loss = this_epoch_data.val_mse
      torch.save(model.state_dict(), model_save_name)
      best_valid_epoch = epoch

    if best_valid_epoch < (epoch - patience):
      break


if __name__ == "__main__":
  main()
