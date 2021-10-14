import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertForSequenceClassification, BertConfig
from contextlib import nullcontext

# Some string constants
CAT, TT, BIN, REG = "cat 2t bin reg".split()
TASKS = [BIN, REG]  # Binary classification and regression
REPS = [CAT, TT]  # Concatenated and two-tower

# BERT constants
BERT_SIZE = 768


class TokenizerMetadata(object):

  def __init__(self, tokenizer):
    self.tokenizer = tokenizer
    self.init_token_idx = self.get_index(tokenizer.cls_token)
    self.eos_token_idx = self.get_index(tokenizer.sep_token)
    self.pad_token_idx = self.get_index(tokenizer.pad_token)
    self.unk_token_idx = self.get_index(tokenizer.unk_token)
    self.max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

  def get_index(self, token):
    return self.tokenizer.convert_tokens_to_ids(token)


def tokenize_and_cut(tokenizer, cut_idx, sentence):
  tokens = tokenizer.tokenize(sentence)
  tokens = tokens[:cut_idx]
  return tokens


def generate_text_field(tokenizer):
  metadata = TokenizerMetadata(tokenizer)
  return data.Field(use_vocab=False,
                    batch_first=True,
                    tokenize=lambda x: tokenize_and_cut(
                        tokenizer, metadata.max_input_length - 2, x),
                    preprocessing=tokenizer.convert_tokens_to_ids,
                    init_token=metadata.init_token_idx,
                    eos_token=metadata.eos_token_idx,
                    pad_token=metadata.pad_token_idx,
                    unk_token=metadata.unk_token_idx)


class DatasetTools(object):

  def __init__(self, tokenizer, device, metadata, fields):
    self.tokenizer = tokenizer
    self.device = device
    self.metadata = metadata
    self.fields = fields


def get_pickle_name(repr_choice, task_choice):
  return "".join(["checkpoints/", repr_choice, "_", task_choice, ".pickle"])


def get_checkpoint_name(repr_choice, task_choice):
  return "".join(["checkpoints/", repr_choice, "_", task_choice, ".pt"])


def calculate_f1(score_map):
  category_accumulators = collections.Counter(
      (a, b) for a, _, b in score_map.values())
  tp = category_accumulators[(1, 1)]
  fp = category_accumulators[(0, 1)]
  fn = category_accumulators[(1, 0)]

  print(category_accumulators)

  if not tp or not fp or not fn:
    return 0.0

  p = tp / (tp + fp)
  r = tp / (tp + fn)

  f1 = 2 * p * r / (p + r)
  return f1


def unpack_key(key):
  a, b = key.split("_")
  return int(a), int(b)


def mean(l):
  if not l:
    return None
  else:
    return sum(l) / len(l)


def calculate_mrr(score_map, dataset_metadata):
  index_to_review_map, review_to_map_map = dataset_metadata

  reviews_represented = set()
  for index, scores in score_map.items():
    reviews_represented.add(index_to_review_map[str(index)])
  mrrs = []
  for review in reviews_represented:
    examples_map = review_to_map_map[review]
    values_lists = collections.defaultdict(list)
    for key, index in review_to_map_map[review].items():
      pred, score, label = score_map[index]
      review_index, rebuttal_index = unpack_key(key)
      values_lists[rebuttal_index].append((rebuttal_index, pred, label))

    for query_index, score_list in values_lists.items():
      reciprocal_ranks = []
      for i, (_, pred, label) in enumerate(
          sorted(score_list, key=lambda x: x[1], reverse=True)):
        if label == 1:
          reciprocal_ranks.append(1 / (i + 1))

      if reciprocal_ranks:
        mrrs.append(mean(reciprocal_ranks))

  return mean(mrrs)


def report_epoch(epoch,
                 task,
                 epoch_data,
                 experiment,
                 dataset_metadata,
                 sub_epoch=0):

  assert task in TASKS

  experiment.log_metric("Epoch train metric",
                        epoch_data.train_metric,
                        step=epoch)
  experiment.log_metric("Epoch val metric", epoch_data.val_metric, step=epoch)

  if task == BIN:
    metric_name = "F1"
    metric_fn = calculate_f1
  else:
    metric_name = "MRR"
    metric_fn = lambda x: calculate_mrr(x, dataset_metadata)

  experiment.log_metric("Epoch train {0}".format(metric_name),
                        metric_fn(epoch_data.train_score_map),
                        step=epoch)
  experiment.log_metric("Epoch dev {0}".format(metric_name),
                        metric_fn(epoch_data.valid_score_map),
                        step=epoch)

  print((
      f'Epoch: {epoch+1:02} {sub_epoch+1:02} | Epoch Time: {epoch_data.elapsed_mins} '
      f'{epoch_data.elapsed_secs}s\n'
      f'\tTrain metric: {epoch_data.train_metric:.3f} | '
      f'\t Val. metric: {epoch_data.val_metric:.3f}'))


def train_or_evaluate(model, iterator, mode, optimizer=None):
  assert mode in "train evaluate".split()

  is_train = mode == "train"

  epoch_loss_sum = 0.0
  score_map = {}

  if is_train:
    model.train()
    context = nullcontext()
    assert optimizer is not None
  else:
    model.eval()
    context = torch.no_grad()

  with context:
    for i, batch in enumerate(iterator):

      mean_loss, predictions = model(batch)

      if is_train:
        optimizer.zero_grad()
        mean_loss.backward()
        optimizer.step()

      else:
        for pred, index, score, label in zip(predictions, batch.overall_index,
                                             batch.score, batch.label):
          score_map[index] = (pred.item(), score.item(), label.item())

      epoch_loss_sum += mean_loss.item() * len(predictions)

  return epoch_loss_sum, score_map


class EpochData(object):

  def __init__(self,
               start_time=None,
               end_time=None,
               metric_dict=None,
               score_maps=None):

    if 'train' in score_maps:
      self.train_metric = metric_dict["train"] / len(score_maps["train"])
      self.train_score_map = score_maps["train"]
    else:
      self.train_metric = -1.0
      self.train_score_map = {}

    if 'dev' in score_maps:
      self.val_metric = metric_dict["dev"] / len(score_maps["dev"])
      self.valid_score_map = score_maps["dev"]
    else:
      self.val_metric = -1.0
      self.valid_score_map = {}

    if 'test' in score_maps:
      self.test_metric = metric_dict["test"] / len(score_maps["test"])
      self.test_score_map = score_maps["test"]
    else:
      self.test_metric = -1.0
      self.test_score_map = {}

    elapsed_time = end_time - start_time
    self.elapsed_mins = int(elapsed_time / 60)
    self.elapsed_secs = int(elapsed_time - (self.elapsed_mins * 60))


CE_LOSS = nn.CrossEntropyLoss()
MSE_LOSS = nn.MSELoss()


def get_a_bert(bert_config):
  return BertModel.from_pretrained('bert-base-uncased', config=bert_config)


class BERTAlignmentModel(nn.Module):

  def __init__(self, repr_type, task_type):

    super().__init__()

    assert repr_type in REPS
    assert task_type in TASKS

    self.repr_type = repr_type
    self.task_type = task_type

    if self.task_type == BIN:
      self.loss_fn = CE_LOSS
      self.label_getter = lambda x: x.label
      num_labels = 2
    else:
      self.loss_fn = MSE_LOSS
      self.label_getter = lambda x: x.score
      num_labels = 1

    bert_uncased_config = config = BertConfig.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
        output_hidden_states=True,
    )

    self.dropout = nn.Dropout(0.25)
    self.classifier = nn.Linear(BERT_SIZE, num_labels)

    if repr_type == CAT:
      self.actual_forward = self._forward_cat
      self.bert = get_a_bert(bert_uncased_config)
    else:
      self.actual_forward = self._forward_2tower
      self.review_bert = get_a_bert(bert_uncased_config)
      self.rebuttal_bert = get_a_bert(bert_uncased_config)
      self.linear = nn.Linear(2 * BERT_SIZE, BERT_SIZE)

    self.checkpoint_name = get_checkpoint_name(repr_type, task_type)

  def forward(self, batch):
    return self._ACTUAL_FORWARD_MAP[self.repr_type](self, batch)

  def _forward_cat(self, batch):
    bert_output = self.bert(batch.both_sentences)
    logits = self.classifier(self.dropout(bert_output[0][:, 0]))
    if self.task_type == REG:
      logits = torch.reshape(logits, [logits.shape[0]])
    loss = self.loss_fn(logits, self.label_getter(batch))
    return loss, self._get_predictions(logits)

  def _forward_2tower(self, batch):
    review_rep = self.review_bert(batch.review_sentence).hidden_states[0][:, 0]
    rebuttal_rep = self.rebuttal_bert(
        batch.rebuttal_sentence).hidden_states[0][:, 0]

    concat_rep = torch.cat([review_rep, rebuttal_rep], 1)
    logits = self.classifier(self.dropout(self.linear(concat_rep)))
    if self.task_type == REG:
      logits = torch.reshape(logits, [logits.shape[0]])
    return self.loss_fn(logits,
                        self.label_getter(batch)), self._get_predictions(logits)

  def _get_predictions(self, logits):
    if self.task_type == REG:
      return logits
    else:
      return torch.argmax(logits, axis=1)

  _ACTUAL_FORWARD_MAP = {CAT: _forward_cat, TT: _forward_2tower}
