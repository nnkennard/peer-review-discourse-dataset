"""Generate politeness labels using Convokit."""
import argparse
import json
from convokit import Speaker, Utterance, Corpus, TextParser, PolitenessStrategies

parser = argparse.ArgumentParser(
description='Generate politeness labels using Convokit.')
parser.add_argument('-i', '--input_file', type=str, help='Input JSON file')

TEXT_PARSER = TextParser()
POLITENESS_STRATEGIES = PolitenessStrategies()

PLACEHOLDER_SPEAKERS = [
Speaker(id="reviewer_id0", name="Reviewer0"),
Speaker(id="reviewer_id1", name="Reviewer1")
  ] # Not really relevant in single-speaker context

def get_convokit_politeness_labels(input_file, use_rebuttal=False):
  """Produce Convokit politeness labels for a review or rebuttal.

    Converts a peer review or rebuttal in the format of this dataset.

    Args:
      input_file: Path to json file for review-rebuttal pair.
      use_rebuttal: If True, process rebuttal sentences; otherwise process
      review sentences.

    Returns:
      A dictionary with review_id as single key, with politeness strategies
      feature dict as its value.
  """

  with open(input_file, 'r') as f:
    obj = json.load(f)
    review_id = obj["metadata"]["review_id"]
    if use_rebuttal:
      relevant_sentences = obj["rebuttal_sentences"]
    else:
      relevant_sentences = obj["review_sentences"]

    corpus = Corpus(
      utterances=[Utterance(text=sentence["text"],
      speaker=PLACEHOLDER_SPEAKERS[0]) for sentence in relevant_sentences])
    corpus = TEXT_PARSER.transform(corpus)
    corpus = POLITENESS_STRATEGIES.transform(corpus, markers=True)
    return {review_id: corpus.get_utterances_dataframe()[
    "meta.politeness_strategies"][0]}


def main():
  args = parser.parse_args()
  print(get_convokit_politeness_labels(args.input_file, True))


if __name__ == "__main__":
  main()

