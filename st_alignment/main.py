from torch.utils.data import DataLoader
from sentence_transformers import losses, util
from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import csv
import os
from tqdm import tqdm
import json
from zipfile import ZipFile
import random
from data_utils import read_data
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout

model = SentenceTransformer('all-MiniLM-L6-v2')
model_save_path = 'output/training_OnlineConstrativeLoss-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(model_save_path, exist_ok=True)
# Training for multiple epochs can be beneficial, as in each epoch a mini-batch is sampled differently
# hence, we get different negatives for each positive
num_epochs = 10

# Increasing the batch size improves the performance for MultipleNegativesRankingLoss. Choose it as large as possible
# I achieved the good results with a batch size of 300-350 (requires about 30 GB of GPU memory)
train_batch_size = 256
# As distance metric, we use cosine distance (cosine_distance = 1-cosine_similarity)
distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE

# Negative pairs should have a distance of at least 0.5
margin = 5

data_dir = "../data_prep/final_dataset/train"
sentence_pairs = read_data(data_dir)
train_samples = [InputExample(texts=[s1, s2], label=label) for (s1, s2, label) in sentence_pairs]
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.OnlineContrastiveLoss(model=model, distance_metric=distance_metric, margin=margin)

dev_data_dir = "/mnt/nfs/scratch1/nnayak/peer-review-discourse-dataset/combined_alignment/torchtext_input_data_posneg_1_sample_1.0/dev/"
dev_sentences1 = []
dev_sentences2 = []
dev_labels = []
evaluators = []
input_files = [f_name for f_name in os.listdir(os.path.join(dev_data_dir)) if f_name.endswith('.jsonl')]
for f_name in input_files:
    dev_file = os.path.join(dev_data_dir, f_name)
    with open(dev_file) as fin:
        for line in tqdm(fin):
            line = json.loads(line.strip())
            review_snt = line["review_sentence"]
            rebuttal_snt = line["rebuttal_sentence"]
            label = line["label"]
            dev_sentences1.append(rebuttal_snt)
            dev_sentences2.append(review_snt)
            dev_labels.append(label)

binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels)
evaluators.append(binary_acc_evaluator)

# Create a SequentialEvaluator. This SequentialEvaluator runs all three evaluators in a sequential order.
# We optimize the model with respect to the score from the last evaluator (scores[-1])
seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])
logger.info("Evaluate model without training")
seq_evaluator(model, epoch=0, steps=0, output_path=model_save_path)


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=seq_evaluator,
          epochs=num_epochs,
          warmup_steps=1000,
          output_path=model_save_path
          )
