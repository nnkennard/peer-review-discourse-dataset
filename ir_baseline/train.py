import argparse
import random

from comet_ml import Experiment

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from contextlib import nullcontext
from torchtext.legacy import data
from transformers import BertTokenizer

import classification_lib

parser = argparse.ArgumentParser(description='prepare CSVs for ws training')
parser.add_argument('-i',
                    '--input_dir',
                    default="score_model_input/",
                    type=str,
                    help='path to data file containing score jsons')
parser.add_argument('-d',
                    '--debug',
                    action='store_true')

def generate_text_field(tokenizer):
  metadata = classification_lib.TokenizerMetadata(tokenizer)
  return data.Field(use_vocab=False,
                    batch_first=True,
                    tokenize=lambda x: classification_lib.tokenize_and_cut(
                        tokenizer, metadata.max_input_length - 2, x),
                    preprocessing=tokenizer.convert_tokens_to_ids,
                    init_token=metadata.init_token_idx,
                    eos_token=metadata.eos_token_idx,
                    pad_token=metadata.pad_token_idx,
                    unk_token=metadata.unk_token_idx)

def make_fields(tokenizer):
  RAW = data.RawField()
  TEXT = generate_text_field(tokenizer)
  LABEL = data.Field(sequential=False, use_vocab=False, dtype=torch.float)

  return {
      "index": ("index", RAW),
      "text": ("text", TEXT),
      "label": ("label", LABEL),
  }

# Setting random seeds
SEED = 43
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def get_dataset_tools(data_dir):
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  metadata = classification_lib.TokenizerMetadata(tokenizer)
  fields = make_fields(tokenizer)
  return classification_lib.DatasetTools(tokenizer, device, metadata, fields)


def build_iterators(data_dir,
                    dataset_tools,
                    batch_size,
                    debug=False,
                    make_valid=False):

  if debug:
    train_file_name = "train_head.jsonl"
  else:
    train_file_name = "train.jsonl"
  train_dataset, dev_dataset = data.TabularDataset.splits(
        path=data_dir,
        train=train_file_name,
        validation=train_file_name.replace("train", "dev"),
        format='json',
        skip_header=True,
        fields=dataset_tools.fields)

  dataset_tools.fields["text"][1].build_vocab(train_dataset)

  return data.BucketIterator.splits([train_dataset, dev_dataset],
                                    batch_size=batch_size,
                                    device=dataset_tools.device,
                                    sort_key=lambda x: x.index,
                                    sort_within_batch=False)

def main():

  args = parser.parse_args()

  dataset_tools = get_dataset_tools(args.input_dir)

  model = classification_lib.BERTRegresser(
      dataset_tools.device).to(dataset_tools.device)
  optimizer = optim.Adam(model.parameters())

  model_save_name = classification_lib.get_checkpoint_name()

  experiment = Experiment(project_name='ir_rank_model')

  all_train_iterator, all_valid_iterator, = build_iterators(
      args.input_dir,
      dataset_tools,
      classification_lib.Hyperparams.batch_size,
      debug=args.debug,
      make_valid=True)

  best_valid_loss = float('inf')
  best_valid_epoch = None

  patience = 5000

  for epoch in range(10000):
    this_epoch_data = classification_lib.do_epoch(
        model, all_train_iterator, optimizer, all_valid_iterator)

    this_epoch_data = classification_lib.do_epoch(model,
                                                  all_train_iterator,
                                                  optimizer,
                                                  all_valid_iterator,
                                                  eval_both=True)
    classification_lib.report_epoch(epoch, this_epoch_data)

    experiment.log_metric("Batch train MSE", this_epoch_data.train_mse,
    step=epoch)
    experiment.log_metric("Batch val MSE", this_epoch_data.val_mse,
    step=epoch)

    if this_epoch_data.val_mse < best_valid_loss:
      print("Best validation loss; saving model from epoch ", epoch)
      best_valid_loss = this_epoch_data.val_mse
      torch.save(model.state_dict(), model_save_name)
      best_valid_epoch = epoch

    if best_valid_epoch < (epoch - patience):
      break


if __name__ == "__main__":
  main()

