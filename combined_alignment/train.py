import argparse
import collections
import glob
import json
import pickle
import time

from comet_ml import Experiment

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from transformers import BertTokenizer
from torchtext.legacy import data

import alignment_lib

parser = argparse.ArgumentParser(description='prepare CSVs for ws training')
parser.add_argument('-i',
                    '--input_dir',
                    default="torchtext_input_data_posneg_1_sample_1.0",
                    type=str,
                    help='path to data file containing score jsons')
parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument(
    '-r',
    '--repr_choice',
    default="cat",
    type=str,
    help='which representation, 2 tower or concatenated (2t/cat)')
parser.add_argument(
    '-t',
    '--task_choice',
    default="bin",
    type=str,
    help='which task, binary classification or regression (bin/reg)')

# Setting random seeds
SEED = 43
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

BATCH_SIZE = 128
EPOCHS = 50
PATIENCE = 5


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


def get_dataset_tools(data_dir):
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  metadata = alignment_lib.TokenizerMetadata(tokenizer)

  field_prep = {
      "overall_index": data.RawField(),
      "review_sentence": generate_text_field(tokenizer),
      "rebuttal_sentence": generate_text_field(tokenizer),
      "both_sentences": generate_text_field(tokenizer),
      "label": data.Field(sequential=False, use_vocab=False, dtype=torch.int64),
      "score": data.Field(sequential=False, use_vocab=False, dtype=torch.float),
  }

  fields = {k: (k, v) for k, v in field_prep.items()}

  return alignment_lib.DatasetTools(tokenizer, device, metadata, fields)


def get_iterator_list(glob_path, debug, dataset_tools):
  iterator_list = []
  filenames = glob.glob(glob_path)
  for filename in tqdm(filenames):
    dataset, = data.TabularDataset.splits(path=".",
                                          train=filename,
                                          format='json',
                                          skip_header=False,
                                          fields=dataset_tools.fields)
    iterator_list.append(
        data.BucketIterator.splits([dataset],
                                   batch_size=BATCH_SIZE,
                                   device=dataset_tools.device,
                                   sort_key=lambda x: x.overall_index,
                                   shuffle=True,
                                   sort_within_batch=False)[0])
    if debug:
      break
  return iterator_list


def build_iterators(data_dir, dataset_tools, debug=False, make_valid=False):

  print("Train iterators")
  train_iterator_list = get_iterator_list(data_dir + "/train/0*.jsonl", debug,
                                          dataset_tools)
  print("Dev iterators")
  dev_iterator_list = get_iterator_list(data_dir + "/dev/0*.jsonl", debug,
                                        dataset_tools)
  print("Test iterators")
  test_iterator_list = get_iterator_list(data_dir + "/test/0*.jsonl", debug,
                                         dataset_tools)

  vocabber_train_dataset, = data.TabularDataset.splits(
      path=data_dir,
      train="vocabber.jsonl",
      format='json',
      skip_header=True,
      fields=dataset_tools.fields)

  for key in "review_sentence rebuttal_sentence both_sentences label".split():
    dataset_tools.fields[key][1].build_vocab(vocabber_train_dataset)

  return train_iterator_list, dev_iterator_list, test_iterator_list


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


def do_epoch(model, iterators, do_train, eval_sets, optimizer=None):

  start_time = time.time()
  if do_train:
    assert optimizer is not None
    print("Train iterators (train)")
    for iterator in tqdm(iterators["train"]):
      _ = alignment_lib.train_or_evaluate(model, iterator, "train", optimizer)

  metric_dict = collections.defaultdict(float)
  score_maps = collections.defaultdict(dict)

  for eval_set in eval_sets:
    assert eval_set in iterators
    for iterator in tqdm(iterators[eval_set]):
      sub_metric, sub_score_map = alignment_lib.train_or_evaluate(
          model, iterator, "evaluate")
      metric_dict[eval_set] += sub_metric
      score_maps[eval_set].update(sub_score_map)

  end_time = time.time()
  return alignment_lib.EpochData(start_time, end_time, metric_dict, score_maps)


def get_metadata(input_dir):
  with open(input_dir + "/metadata.json", 'r') as f:
    obj = json.load(f)
    return obj["index_to_review_map"], obj["review_to_map_map"]


def main():

  args = parser.parse_args()

  dataset_tools = get_dataset_tools(args.input_dir)

  experiment = Experiment(project_name=args.repr_choice + args.task_choice)

  print("Getting iterators")
  train_iterators, dev_iterators, test_iterators = build_iterators(
      args.input_dir, dataset_tools, debug=args.debug, make_valid=True)

  dataset_metadata = get_metadata(args.input_dir)

  model = alignment_lib.BERTAlignmentModel(args.repr_choice, args.task_choice)
  model.to(dataset_tools.device)

  optimizer = optim.Adam(model.parameters(), lr=5e-5)

  best_valid_loss = float('inf')
  best_valid_epoch = None
  for epoch in range(EPOCHS):

    print("Starting epoch {0}".format(epoch))

    do_epoch(model,
             iterators={"train": train_iterators},
             do_train=True,
             eval_sets=[],
             optimizer=optimizer)

    print("Ran train epoch")

    this_epoch_data = do_epoch(model,
                               iterators={
                                   "train": train_iterators,
                                   "dev": dev_iterators
                               },
                               do_train=False,
                               eval_sets=["train", "dev"])

    print("in train.py", args.task_choice)
    alignment_lib.report_epoch(epoch, args.task_choice, this_epoch_data,
                               experiment, dataset_metadata)

    print("Ran eval epoch")

    if this_epoch_data.val_metric < best_valid_loss:
      print("Best validation loss; saving model from epoch ", epoch)
      best_valid_loss = this_epoch_data.val_metric
      best_epoch_data = this_epoch_data
      torch.save(model.state_dict(), model.checkpoint_name)
      best_valid_epoch = epoch

    if best_valid_epoch < (epoch - PATIENCE):
      torch.load(model.checkpoint_name)
      best_epoch_data_with_test = do_epoch(model,
                                           iterators={
                                               "train": train_iterators,
                                               "dev": dev_iterators,
                                               "test": test_iterators
                                           },
                                           do_train=False,
                                           eval_sets=["train", "dev", "test"])
      print("Dumping test results")
      with open(
          alignment_lib.get_pickle_name(args.repr_choice, args.task_choice),
          'wb') as f:
        pickle.dump(best_epoch_data_with_test, f)
      break


if __name__ == "__main__":
  main()
