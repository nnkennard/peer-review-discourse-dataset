import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertConfig
from contextlib import nullcontext


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


def get_checkpoint_name(repr_choice, task_choice):
  return "".join(["checkpoints/", repr_choice, "_", task_choice, ".pt"])


def do_epoch(model,
             train_iterator,
             optimizer,
             valid_iterator,
             eval_both=False,
             dev_only=False):

  start_time = time.time()
  if eval_both:
    train_set_mode = "evaluate"
  else:
    train_set_mode = "train"
  if dev_only:
    train_metric = None
    train_score_map = {}
  if not dev_only:
    train_metric, train_score_map = train_or_evaluate(model, train_iterator,
                                                      train_set_mode, optimizer)
  valid_metric, valid_score_map = train_or_evaluate(model, valid_iterator,
                                                    "evaluate")
  end_time = time.time()
  return EpochData(start_time, end_time, train_metric, valid_metric,
                   train_score_map, valid_score_map)


def report_epoch(epoch, epoch_data, experiment, sub_epoch=0):
  experiment.log_metric("Batch train metric",
                        epoch_data.train_metric,
                        step=epoch)
  experiment.log_metric("Batch val metric", epoch_data.val_metric, step=epoch)

  print((
      f'Epoch: {epoch+1:02} {sub_epoch+1:02} | Epoch Time: {epoch_data.elapsed_mins} '
      f'{epoch_data.elapsed_secs}s\n'
      f'\tTrain MSE: {epoch_data.train_metric:.3f} | '
      f'\t Val. MSE: {epoch_data.val_metric:.3f}'))


def train_or_evaluate(model, iterator, mode, optimizer=None):
  assert mode in "train evaluate".split()

  is_train = mode == "train"

  epoch_loss = 0.0
  epoch_acc = 0.0
  example_counter = 0
  score_map = {}

  if is_train:
    model.train()
    context = nullcontext()
    assert optimizer is not None
  else:
    model.eval()
    context = torch.no_grad()

  with context:
    for batch in iterator:

      example_counter += len(batch)
      if is_train:
        optimizer.zero_grad()

      loss, predictions = model(batch)

      score_map.update({
          index: (pred.item(), score.item(), label.item())
          for index, pred, score, label in zip(batch.overall_index, predictions,
                                               batch.score, batch.label)
      })

      if is_train:
        loss.backward()
        optimizer.step()

      epoch_loss += loss.item()

  assert example_counter
  return epoch_loss / example_counter, score_map


class EpochData(object):

  def __init__(self, start_time, end_time, train_metric, val_metric,
               train_score_map, valid_score_map):
    self.train_metric = train_metric
    self.val_metric = val_metric

    elapsed_time = end_time - start_time
    self.elapsed_mins = int(elapsed_time / 60)
    self.elapsed_secs = int(elapsed_time - (self.elapsed_mins * 60))

    self.train_score_map = train_score_map
    self.valid_score_map = valid_score_map


BCELOSS = nn.BCEWithLogitsLoss()
MSE_LOSS = nn.MSELoss()


def get_a_bert(config_dict):
  config = BertConfig.from_pretrained("bert-base-uncased")
  config.num_labels = config_dict.get("num_labels", 1)
  return BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                       config=config)


class BERTAlignmentModel(nn.Module):

  def __init__(self, repr_type, task_type):

    super().__init__()

    # Set up berts

    self.repr_type = repr_type
    self.task_type = task_type

    if repr_type == "cat":
      self.actual_forward = self._forward_cat
      self.cat_bert = get_a_bert(
          {"num_labels": self._TASK_TO_NUM_LABELS[task_type]})
      self.cat_bert.config.num_labels = self._TASK_TO_NUM_LABELS[task_type]
    else:
      assert repr_type == "2t"
      self.actual_forward = self._forward_2tower
      self.review_bert = get_a_bert({})
      self.rebuttal_bert = get_a_bert({})
      for mod in [self.review_bert, self.rebuttal_bert]:
        mod.config.output_hidden_states = True
      self.dropout = nn.Dropout(0.25)
      self.linear1 = nn.Linear(1536, 768)
      self.classifier = nn.Linear(768, 1)

    self.label_getter = self._LABEL_GETTER_GETTER[repr_type][task_type]
    self.checkpoint_name = get_checkpoint_name(repr_type, task_type)

  def forward(self, batch):
    return self._ACTUAL_FORWARD_MAP[self.repr_type](self, batch)

  def _forward_cat(self, batch):
    output = self.cat_bert(batch.both_sentences,
                           labels=self.label_getter(batch))
    return output.loss, self._get_predictions(output.logits)

  def _forward_2tower(self, batch):
    review_rep = self.review_bert(batch.review_sentence).hidden_states[0][:, 0]
    rebuttal_rep = self.rebuttal_bert(
        batch.rebuttal_sentence).hidden_states[0][:, 0]

    concat_rep = torch.cat([review_rep, rebuttal_rep], 1)
    logits = self.classifier(self.linear1(self.dropout(concat_rep)))

    if self.task_type == 'reg':
      loss = MSE_LOSS(torch.reshape(logits, [len(logits),1]),
      torch.reshape(batch.score, [len(logits), 1]))
    else:
      loss = BCELOSS(torch.reshape(logits, [len(logits),1]), self.label_getter(batch))

    return loss, self._get_predictions(logits)

  def _get_predictions(self, logits):
    if self.task_type == 'bin' and self.repr_type == 'cat':
      return torch.argmax(logits, axis=1)
    else:
      return logits

  _TASK_TO_NUM_LABELS = {"bin": 2, "reg": 1}

  _ACTUAL_FORWARD_MAP = {"cat": _forward_cat, "2t": _forward_2tower}

  _LABEL_GETTER_GETTER = {
      "cat": {
          "bin": lambda x: x.label,
          "reg": lambda x: x.score
      },
      "2t": {
          "bin": lambda x: torch.reshape(x.label.float(), [len(x.label), 1]),
          "reg": lambda x: x.score
      },
  }
