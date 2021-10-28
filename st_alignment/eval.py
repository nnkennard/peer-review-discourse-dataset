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

logger.info("Loading model...")
model = SentenceTransformer('output/training_OnlineConstrativeLoss-2021-10-16_01-04-58')  # model trained with contrastive and without NO_MATCH; this model was also trained on the preprocessed data given by Neha here /home/rajarshi/Dropbox/research/peer-review-discourse-dataset/data_prep/final_dataset/train
# model = SentenceTransformer('output/training_OnlineConstrativeLoss-2021-10-16_03-34-57')  # model trained with ranking loss and NO_MATCH
# model = SentenceTransformer('output/training_OnlineConstrativeLoss-2021-10-16_03-52-04')  # model trained with contrastive and NO_MATCH
# model = SentenceTransformer('all-MiniLM-L6-v2')  # model trained with contrastive and NO_MATCH

data_dir = "/home/rajarshi/Dropbox/research/peer-review-discourse-dataset/data_prep/final_dataset/"
input_files = [f_name for f_name in os.listdir(os.path.join(data_dir, "test")) if f_name.endswith('.json')]
all_rr = []
for f in tqdm(input_files):
    with open(os.path.join(data_dir, "test", f)) as fin:
        data = json.load(fin)
    rebuttal_sentences_text = [t["text"] for t in data["rebuttal_sentences"]]
    review_sentences_text = [t["text"] for t in data["review_sentences"]] + ["NO_MATCH"]
    rebuttal_sentences_emb = model.encode(rebuttal_sentences_text)
    review_sentences_emb = model.encode([t["text"] for t in data["review_sentences"]])
    # Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.pytorch_cos_sim(rebuttal_sentences_emb, review_sentences_emb)
    ranks = torch.argsort(-cosine_scores, dim=1)
    review_sentences = data["review_sentences"]
    for ctr, rb_s in enumerate(data["rebuttal_sentences"]):
        rr = 0
        type, alignments = rb_s["alignment"]
        if alignments is None:
            alignments = [len(review_sentences_text) - 1]
        ranks_rbs = ranks[ctr]
        success_ctr = 0  # number of alignments found successfully so far
        for r_ctr, r in enumerate(ranks_rbs):
            if r in alignments:
                rr += (1/(r_ctr + 1 - success_ctr))
                success_ctr += 1
        rr = rr/len(alignments)
        all_rr.append(rr)
logger.info("MRR: {}".format(np.mean(all_rr)))





