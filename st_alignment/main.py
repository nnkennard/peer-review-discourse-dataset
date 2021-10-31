import datetime
import logging
import os
from sentence_transformers import losses, LoggingHandler, SentenceTransformer, evaluation
from sentence_transformers.readers import InputExample


from data_utils import read_data
from eval_lib import eval_dir
from torch.utils.data import DataLoader


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


NUM_EPOCHS = 10
TRAIN_BATCH_SIZE = 256
DISTANCE_METRIC = losses.SiameseDistanceMetric.COSINE_DISTANCE
MARGIN = 5
TRAIN_DATA_DIR = "../data_prep/final_dataset/train"


def create_model():
  model = SentenceTransformer('all-MiniLM-L6-v2')
  model_save_path = 'output/training_OnlineConstrativeLoss-' + datetime.datetime.now(
  ).strftime("%Y-%m-%d_%H-%M-%S")
  os.makedirs(model_save_path, exist_ok=True)
  return model, model_save_path


def main():

  model, model_save_path = create_model()

  train_sentence_pairs = read_data(TRAIN_DATA_DIR)
  train_samples = [
      InputExample(texts=[s1, s2], label=label)
      for (s1, s2, label) in train_sentence_pairs
  ]
  train_dataloader = DataLoader(train_samples,
                                shuffle=True,
                                batch_size=TRAIN_BATCH_SIZE)
  train_loss = losses.OnlineContrastiveLoss(model=model,
                                            distance_metric=DISTANCE_METRIC,
                                            margin=MARGIN)

  dev_sentence_pairs = read_data(TRAIN_DATA_DIR.replace("train", "dev"))
  dev_revs, dev_rebs, dev_labels = zip(*dev_sentence_pairs)

  # We optimize the model with respect to the score from the last evaluator (scores[-1])
  seq_evaluator = evaluation.SequentialEvaluator(
      [
          evaluation.BinaryClassificationEvaluator(dev_revs,
                                                   dev_rebs, dev_labels)
      ],
      main_score_function=lambda scores: scores[-1])
  logger.info("Evaluate model without training")
  seq_evaluator(model, epoch=0, steps=0, output_path=model_save_path)

  # Train the model
  model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=seq_evaluator,
            epochs=NUM_EPOCHS,
            warmup_steps=1000,
            output_path=model_save_path)
  eval_dir(model_save_path)
  eval_dir(model_save_path, subset="test")


if __name__ == "__main__":
  main()
