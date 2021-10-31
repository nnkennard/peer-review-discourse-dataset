import glob
import json
import os
from tqdm import tqdm
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import losses, util
from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation
from sentence_transformers.readers import InputExample

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

#model = SentenceTransformer(
#    'output/training_OnlineConstrativeLoss-2021-10-27_23-46-38'
#)  # model trained with contrastive and without NO_MATCH; this model was also trained on the preprocessed data given by Neha here /home/rajarshi/Dropbox/research/peer-review-discourse-dataset/data_prep/final_dataset/train
# model = SentenceTransformer('output/training_OnlineConstrativeLoss-2021-10-16_03-34-57')  # model trained with ranking loss and NO_MATCH
# model = SentenceTransformer('output/training_OnlineConstrativeLoss-2021-10-16_03-52-04')  # model trained with contrastive and NO_MATCH
# model = SentenceTransformer('all-MiniLM-L6-v2')  # model trained with contrastive and NO_MATCH


def eval_dir(model_save_path,data_dir="../data_prep/final_dataset", subset="dev"):

  glob_path = data_dir + "/" + subset + "/*.json"
  logger.info("Loading model...")
  model = SentenceTransformer(model_save_path)

  all_rr = []
  for filename in glob.glob(glob_path):
    with open(filename) as fin:
      data = json.load(fin)
    rebuttal_sentences_text = [t["text"] for t in data["rebuttal_sentences"]]
    review_sentences_text = [t["text"] for t in data["review_sentences"]
                            ] + ["NO_MATCH"]
    rebuttal_sentences_emb = model.encode(rebuttal_sentences_text)
    review_sentences_emb = model.encode(
review_sentences_text)
    # Why does NO_MATCH not get encoded?
    # Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.pytorch_cos_sim(rebuttal_sentences_emb,
                                         review_sentences_emb)
    ranks = torch.argsort(-cosine_scores, dim=1)
    for ctr, rb_s in enumerate(data["rebuttal_sentences"]):
      rr = 0
      _, alignments = rb_s["alignment"]
      if alignments is None:
        alignments = [len(review_sentences_text) - 1]
      ranks_rbs = ranks[ctr]
      success_ctr = 0  # number of alignments found successfully so far
      for r_ctr, r in enumerate(ranks_rbs):
        if r in alignments:
          rr += (1 / (r_ctr + 1 - success_ctr))
          success_ctr += 1
      rr = rr / len(alignments)
      all_rr.append(rr)
  logger.info("{} MRR: {}".format(subset, np.mean(all_rr)))
