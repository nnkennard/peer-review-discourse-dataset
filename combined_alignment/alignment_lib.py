import time
import torch
import torch.nn as nn
#from transformers import BertTokenizer
from transformers import BertModel
from contextlib import nullcontext


class Hyperparams(object):
  hidden_dim = 512
  output_dim = 2
  n_layers = 2
  bidirectional = True
  dropout = 0.25
  batch_size = 128


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
             loss,
             label_getter,
             eval_both=False,
             dev_only=False):

  start_time = time.time()
  if eval_both:
    train_set_mode = "evaluate"
  else:
    train_set_mode = "train"
  if dev_only:
    train_mse = 0; train_score_map ={}
  if not dev_only:
    train_mse, train_score_map = train_or_evaluate(model, train_iterator,
                                                   train_set_mode, loss,
                                                   label_getter,
                                                   optimizer)
  valid_mse, valid_score_map = train_or_evaluate(model, valid_iterator,
                                                 "evaluate", loss, label_getter)
  end_time = time.time()
  return EpochData(start_time, end_time, train_mse, valid_mse, train_score_map,
                   valid_score_map)


def report_epoch(epoch, epoch_data, experiment, sub_epoch=0):
  experiment.log_metric("Batch train metric",
                            epoch_data.train_mse,
                            step=epoch)
  experiment.log_metric("Batch val metric", epoch_data.val_mse, step=epoch)


  print((f'Epoch: {epoch+1:02} {sub_epoch+1:02} | Epoch Time: {epoch_data.elapsed_mins} '
         f'{epoch_data.elapsed_secs}s\n'
         f'\tTrain MSE: {epoch_data.train_mse:.3f} | '
         f'\t Val. MSE: {epoch_data.val_mse:.3f}'))


def train_or_evaluate(model, iterator, mode, loss_fn, label_getter, optimizer=None):
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

  #if type(loss) == nn.BCEWithLogitsLoss:
  #  pass
    


  with context:
    for batch in iterator:

      example_counter += len(batch)
      if is_train:
        optimizer.zero_grad()

      predictions = model(batch).squeeze(1)
      loss = loss_fn(predictions, label_getter(batch))


      score_map.update({
          index: score.item()
          for index, score in zip(batch.overall_index, predictions)
      })

      if is_train:
        loss.backward()
        optimizer.step()

      epoch_loss += loss.item() * len(predictions)

  assert example_counter
  return epoch_loss / example_counter, score_map


class EpochData(object):

  def __init__(self, start_time, end_time, train_mse, val_mse, train_score_map,
               valid_score_map):
    self.train_mse = train_mse
    self.val_mse = val_mse

    elapsed_time = end_time - start_time
    self.elapsed_mins = int(elapsed_time / 60)
    self.elapsed_secs = int(elapsed_time - (self.elapsed_mins * 60))

    self.train_score_map = train_score_map
    self.valid_score_map = valid_score_map


class BERTAlignmentModel(nn.Module):

  def __init__(self, repr_type):

    super().__init__()

    if repr_type == "cat":
      self.actual_forward = self._forward_cat
    else:
      assert repr_type == "2t"
      self.actual_forward = self._forward_2tower

    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.dropout = nn.Dropout(Hyperparams.dropout)
    self.out = nn.Linear(768, 1)

  def forward(self, batch):
    return self.actual_forward(batch)

  def _forward_cat(self, batch):
    with torch.no_grad():
      embedded = self.bert(batch.both_sentences)[0][:, 0]
    return self.out(self.dropout(embedded))

  def _forward_2tower(self, batch):
    with torch.no_grad():
      embedded_review = self.bert(batch.review_sentence)[0][:, 0]
      embedded_rebuttal = self.bert(batch.rebuttal_sentence)[0][:, 0]
    return self.out(self.dropout(embedded_review))
