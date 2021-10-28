import json
import os
from tqdm import tqdm
import numpy as np
def read_data(data_dir):

    input_files = [f_name for f_name in os.listdir(os.path.join(data_dir)) if f_name.endswith('.json')]
    sentence_pairs = []
    for f in tqdm(input_files):
        with open(os.path.join(data_dir, f)) as fin:
            data = json.load(fin)
        rebuttal_sentences_text = [t["text"] for t in data["rebuttal_sentences"]]
        review_sentences_text = [t["text"] for t in data["review_sentences"]]
        for ctr, rb_s in enumerate(data["rebuttal_sentences"]):
            flag = 0
            type, alignments = rb_s["alignment"]
            if alignments is not None:
                for a_ind in alignments:
                    if a_ind >= len(review_sentences_text):
                        flag = 1
                        break
            if flag == 1:
                continue
            if alignments is None:
                sentence_pairs.append((rebuttal_sentences_text[ctr], "NO_MATCH", 1))
                # put some negative examples
                negative_ind = np.random.randint(0, len(review_sentences_text))
                sentence_pairs.append((rebuttal_sentences_text[ctr], review_sentences_text[negative_ind], 0))
            else:
                for a_ind in alignments:
                    sentence_pairs.append((rebuttal_sentences_text[ctr], review_sentences_text[a_ind], 1))
                    # put some negative examples
                    negative_ind = a_ind
                    while negative_ind in alignments:
                        negative_ind = np.random.randint(0, len(review_sentences_text))
                    if negative_ind not in alignments:
                        sentence_pairs.append((rebuttal_sentences_text[ctr], review_sentences_text[negative_ind], 0))
    return sentence_pairs

