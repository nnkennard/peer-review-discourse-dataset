import argparse
from transformers import BertTokenizer
from transformers import BertModel
import numpy as np
import glob
import time
from contextlib import nullcontext
import random
import torch
import torch.optim as optim
from torchtext.legacy import data
import torch.nn as nn
from tqdm import tqdm

import classification_lib

parser = argparse.ArgumentParser(description='prepare CSVs for ws training')
parser.add_argument('-d',
                    '--datadir',
                    default="data/ws/",
                    type=str,
                    help='path to data file containing score jsons')
parser.add_argument('-m',
                    '--model',
                    default="rankprob",
                    type=str,
                    help='which model to train')



def make_fields(model, tokenizer):
  RAW = data.RawField()
  TEXT = generate_text_field(tokenizer)
  if model == classification_lib.Model.rank:
    LABEL = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
  else:
    LABEL = data.LabelField(dtype=torch.long, use_vocab=True)

  return {
      "index": ("index", RAW),
      "text": ("text", TEXT),
      "label": ("label", LABEL),
  }


# Setting random seeds
SEED = 43
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

logit_lookup = np.eye(2)


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


def get_dataset_tools(data_dir, model):
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  metadata = classification_lib.TokenizerMetadata(tokenizer)
  fields = make_fields(model, tokenizer)

  # Create fake train obj

  temp_obj, = data.TabularDataset.splits(path=data_dir,
                                         train=model + '_vocabber.jsonl',
                                         format='json',
                                         fields=fields,
                                         skip_header=True)

  for name, (_, field) in fields.items():
    if name == 'text':
      field.build_vocab(temp_obj)
    elif name == 'label' and model == 'rankprob':
      field.build_vocab(temp_obj)

  return classification_lib.DatasetTools(tokenizer, device, metadata, fields)


def build_iterators(data_dir,
                    train_file_name,
                    dataset_tools,
                    batch_size,
                    make_valid=False):
  if make_valid:
    iterator_input = data.TabularDataset.splits(
        path=data_dir,
        train=train_file_name,
        validation=train_file_name.replace("train_", "dev_"),
        format='json',
        skip_header=True,
        fields=dataset_tools.field_map)
  else:
    iterator_input = data.TabularDataset.splits(path=data_dir,
                                           train=train_file_name,
                                           format='json',
                                           skip_header=True,
                                           fields=dataset_tools.field_map)


  return data.BucketIterator.splits(iterator_input,
                                    batch_size=batch_size,
                                    device=dataset_tools.device,
                                    sort_key=lambda x: x.index,
                                    sort_within_batch=False)

def main():

  args = parser.parse_args()

  assert args.model in classification_lib.Model.ALL
  dataset_tools = get_dataset_tools(args.datadir, args.model)

  model = classification_lib.BERTGRUClassifier(
      dataset_tools.device, args.model).to(dataset_tools.device)
  optimizer = optim.Adam(model.parameters())
  if args.model == classification_lib.Model.rank:
    criterion = nn.MSELoss()
  else:
    criterion = nn.CrossEntropyLoss()

  model_save_name  = classification_lib.get_checkpoint_name(args.model)


  all_train_iterator, all_valid_iterator, = build_iterators(
      args.datadir,
      "train_" + args.model + "_batch_0.jsonl",
      dataset_tools,
      classification_lib.Hyperparams.batch_size,
      make_valid=True)

  best_valid_loss = float('inf')
  best_valid_epoch = None

  patience = 5

  glob_getter = args.datadir + "/train_" + args.model + "_*.jsonl"

  metric_map = {
      "rank": classification_lib.my_mse,
      "rankprob": classification_lib.binary_accuracy
  }

  for epoch in range(100):
    for train_file in tqdm(sorted(glob.glob(glob_getter))):
      if 'all' in train_file:
        continue
      train_file_name = train_file.split('/')[-1]

      train_iterator, = build_iterators(
          args.datadir, train_file_name, dataset_tools,
          classification_lib.Hyperparams.batch_size, make_valid=False)

      this_epoch_data = classification_lib.do_epoch(
          model, train_iterator, criterion, metric_map[args.model],
          lambda x: x.label, 2, optimizer, all_valid_iterator)

    this_epoch_data = classification_lib.do_epoch(model,
                                                  all_train_iterator,
                                                  criterion,
                                                  metric_map[args.model],
                                                  lambda x: x.label,
                                                  2,
                                                  optimizer,
                                                  all_valid_iterator,
                                                  eval_both=True)
    classification_lib.report_epoch(epoch, this_epoch_data)

    if this_epoch_data.val_loss < best_valid_loss:
      print("Best validation loss; saving model from epoch ", epoch)
      best_valid_loss = this_epoch_data.val_loss
      torch.save(model.state_dict(), model_save_name)
      best_valid_epoch = epoch

    if best_valid_epoch < (epoch - patience):
      break


if __name__ == "__main__":
  main()

