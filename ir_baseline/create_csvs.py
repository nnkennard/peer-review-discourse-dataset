import argparse
import collections
import sys

parser = argparse.ArgumentParser(description='prepare CSVs for ws training')
parser.add_argument('-d',
                    '--datadir',
                    default="../0517_split_2/train/",
                    type=str,
                    help='path to data file containing score jsons')
parser.add_argument('-m',
                    '--model',
                    default="rankprob",
                    type=str,
                    help='which IR model to prepare for')


POS = "1_more_relevant"
NEG = "2_more_relevant"
FIELDS = "id label text".split()
RowTuple = collections.namedtuple('RowTuple', FIELDS)

PAIRS_PER_FILE = 20 # What actually is this?

class Model(object):
  rank = 'rank'
  rankprob = 'rankprob'
  ALL = [rank, rankprob]

def rank_process(offset, rebuttal_sentence, review_sentences):
  examples =  []
  for i, (review_sentence, score) in enumerate(review_sentences):
    examples.append(
        RowTuple(offset + i, ' [SEP] '.join(
            [rebuttal_sentence, review_sentence]), score)._asdict())
  return examples


def rankprob_process(offset, rebuttal_sentence, review_sentences):
  examples =  []
  d1s = review_sentences[:SAMPLES_PER_SENTENCE[Models.rankprob]/2]
  d2s = review_sentences[SAMPLES_PER_SENTENCE[Models.rankprob]/2:]
  index = offset

  for d1, score_1 in d1s:
    for d2, score_2 in d2s:
      if score_1 > score_2:
        label = POS
      else:
        label = NEG
      examples.append(
      RowTuple(
        index, ' [SEP] '.join([rebuttal_sentence, d1, d2]), label
      )._asdict())
      index += 1
  return examples


FUNC_MAP = {
  Model.rank: rank_process,
  Model.rankprob : rankprob_process
}

SAMPLES_PER_SENTENCE = {
  Models.rank: 10,
  Models.rankprob: 20
}

def get_samples(filename, process_fn, samples_per_sentence):
  with open(filename, 'r') as f:
    obj = json.load(f)
  # for each rebuttal sentence
    # sample some review sentences
    # get the scores
    # create torchtext examples




def main():

  args = parser.parse_args()
  assert args.model in Model.ALL


  for subset in "train dev".split():

    all_filenames = list(sorted(glob.glob("/".join(
        [args.datadir, subset, "*.json"]))))

    for batch_i, input_file_start_index in enumerate(
      tqdm(range(0, len(all_filenames), PAIRS_PER_FILE))):

      this_file_filenames = all_filenames[
          input_file_start_index:input_file_start_index + PAIRS_PER_FILE]

      sample_collector = []
      for filename in this_file_filenames:
        sample_collector += get_samples(filename, FUNC_MAP[args.model])

      sample_builder = {"train":[], "dev":[]}
      for filename in this_file_filenames:
        subset = "train" if bernoulli(TRAIN_FRAC) else "dev"
        with open(filename, 'r') as f:
          data_obj = json.load(f)

        overall_builders["d"] += data_obj["review"]
        overall_builders["q"] += data_obj["rebuttal"]
        sample_builder[subset] += sample_things(data_obj)

      for subset, examples in sample_builder.items():
        output_file = "".join([
          args.datadir, "batch_", str(batch_i), "_", subset,  ".csv"])
        with open(output_file, 'w') as g:
          writer = csv.DictWriter(g, FIELDS)
          writer.writeheader()
          for i, example in enumerate(examples):
            writer.writerow(example)


    fake_sentences = {}
    for builder_type in ["d", "q"]:
      fake_sentences[builder_type] = build_overall_vocab(
        overall_builders[builder_type])


    stop_len = max([len(fake_sentences["q"]), len(fake_sentences["d"])])
    with open(args.datadir + "/overall_dummy_vocabber.csv", 'w') as h:
      writer = csv.DictWriter(h, FIELDS)
      writer.writeheader()
      for i, (d1, d2, q, label) in enumerate(
          zip(*[
              itertools.cycle(x) for x in [
                  fake_sentences["d"], fake_sentences["d"],
                  fake_sentences["q"], [POS, NEG]
              ]
          ])):
        if i == stop_len:
          break
        writer.writerow(
            {"id": i,
              "text": " [SEP] ".join([d1, d2, q]),
              "label": label})


if __name__ == "__main__":
  pass
