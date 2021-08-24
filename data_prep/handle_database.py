import argparse
import collections
import json
import logging

import data_prep_lib as dpl
from data_prep_lib import AnnotationFields as FIELDS

parser = argparse.ArgumentParser(
    description=
    "Convert the outputs form the annotation server into a cleaned dataset.")
parser.add_argument('-t',
                    '--text_dump',
                    default="../final_data_dump/orda_text_0415.json",
                    type=str,
                    help='path to text dump from annotation server')
parser.add_argument(
    '-a',
    '--annotation_dump',
    #default="../final_data_dump/orda_annotations_0516.json",
    default="../final_data_dump/orda_annotations_0820.json",
    type=str,
    help='path to annotation dump from annotation server')
parser.add_argument('-i',
                    '--intermediate_file',
                    default="filtered_database.json",
                    type=str,
                    help='where to dump intermediate collated data items')

FrozenAnnotationCollection = collections.namedtuple(
    "FrozenAnnotationCollection",
    "review_id annotator review_annotation review_sentences rebuttal_sentences".
    split())


class AnnotationTypes(object):
  rev_ann = "reviewannotation"
  rev_sent_ann = "reviewsentenceannotation"
  reb_sent_ann = "rebuttalsentenceannotation"
  ALL = [rev_ann, rev_sent_ann, reb_sent_ann]


TYPE_TO_KEY_MAP = {
    AnnotationTypes.rev_sent_ann: "review_sentence_index",
    AnnotationTypes.reb_sent_ann: "rebuttal_sentence_index",
}

with open('annomap.json', 'r') as f:
  ANONYMIZER = json.load(f)

REVIEW_ID, INITIALS = "review_id initials".split()


def get_key_from_annotation(ann):
  return ann["fields"][REVIEW_ID], ANONYMIZER[ann["fields"][INITIALS]]


def get_text(text_dump_path, review_ids):
  """Get text for all reviews and rebuttals from server dataset dump.

    Args:
      text_dump_path: Path to json dump of text
      review_ids: List of review_ids whose text is required

    Returns:
      comment_pair_map: A map from review ids to details for the
      review-rebuttal pair. Details are in a dict including review text,
      rebuttal text and metadata.
  """
  with open(text_dump_path, 'r') as f:
    j = json.load(f)

  # Builds sentence map for all comments in server database
  sentence_map = collections.defaultdict(list)
  for sentence in j[dpl.ServerModels.sentence]:
    fields = dpl.get_fields(sentence)
    assert fields[FIELDS.sentence_index] == len(
        sentence_map[fields[FIELDS.comment_id]])
    sentence_map[fields[FIELDS.comment_id]].append(fields[FIELDS.text])

  comment_pair_map = {}
  for example in j[dpl.ServerModels.example]:
    fields = dpl.get_fields(example)
    review_id, rebuttal_id = fields[FIELDS.review_id], fields[
        FIELDS.rebuttal_id]
    if review_id not in review_ids:
      continue
    comment_pair_map[review_id] = {
        dpl.REVIEW: sentence_map[review_id],
        dpl.REBUTTAL: sentence_map[rebuttal_id],
        dpl.METADATA: dpl.metadata_formatter(fields)
    }
  return comment_pair_map


class AnnotationCollector(object):

  def __init__(self, review_id, annotator):
    self.review_id = review_id
    self.annotator = annotator
    self.annotations = {
        annotation_type: [] for annotation_type in AnnotationTypes.ALL
    }

  def is_valid(self):
    if self.review_id == 'example_review':
      logging.info("example_review {0} -- skipping".format(self.annotator))
      return False
    for annotation_type, annotation_list in self.annotations.items():
      if not annotation_list:
        logging.info("{0} {1} -- no annotations of type {2}".format(
            self.review_id, self.annotator, annotation_type))
        return False
    if len(self.annotations[AnnotationTypes.rev_ann]) > 1:
      return False
    return True

  def get_review_annotation(self):
    return sorted(self.annotations[AnnotationTypes.rev_ann],
                  key=lambda x: x["pk"])[-1]

  def filter_annotations_for_latest(self, annotation_type):
    final_annotations = {}
    key = TYPE_TO_KEY_MAP[annotation_type]
    for annotation in sorted(self.annotations[annotation_type],
                             key=lambda x: x["pk"]):
      final_annotations[annotation["fields"][key]] = annotation
    return tuple(final_annotations[k] for k in sorted(final_annotations.keys()))

  def _remap_review_sentences(self, review_sentences):
    sentence_map = {}
    for sentence in review_sentences:
      all_together_dict = sentence["fields"]
      maybe_labels = dpl.recursive_json_load(all_together_dict["labels"])
      if "0" not in maybe_labels:
        #raise dpl.NoLabelError
        pass
      else:
        all_together_dict.update(maybe_labels["0"])
      sentence_map[all_together_dict["review_sentence_index"]] = all_together_dict
    print(sentence_map.keys())
    return sentence_map

  def freeze(self):
    if self.is_valid():
      annotation_map_builder = {
          k: self.filter_annotations_for_latest(k)
          for k in [AnnotationTypes.rev_sent_ann, AnnotationTypes.reb_sent_ann]
      }
      return FrozenAnnotationCollection(
          self.review_id,
          self.annotator,
          self.get_review_annotation(),
          self._remap_review_sentences(annotation_map_builder[AnnotationTypes.rev_sent_ann]),
          annotation_map_builder[AnnotationTypes.reb_sent_ann],
      )._asdict()
    else:
      logging.info("{0} {1} -- some kind of error; returning None".format(
          self.review_id, self.annotator))
      return None


def build_annotation_collectors(annotation_file):
  with open(annotation_file, 'r') as f:
    annotations_from_file = json.load(f)

  annotation_collectors = collections.defaultdict(dict)

  for annot in AnnotationTypes.ALL:
    for row in annotations_from_file[annot]:
      rev_id, annotator = get_key_from_annotation(row)
      # Start a collector if necessary
      if annotator not in annotation_collectors[rev_id]:
        annotation_collectors[rev_id][annotator] = AnnotationCollector(
            rev_id, annotator)
      annotation_collectors[rev_id][annotator].annotations[annot].append(row)

  frozen_annotation_collectors = collections.defaultdict(dict)
  for k1, v1 in annotation_collectors.items():
    for k2, v2 in v1.items():
      frozen_annotation_collectors[k1][k2] = v2.freeze()

  return frozen_annotation_collectors


def main():

  args = parser.parse_args()

  annotation_collectors = build_annotation_collectors(args.annotation_dump)
  comment_pair_map = get_text(args.text_dump, annotation_collectors.keys())

  with open(args.intermediate_file, 'w') as f:
    json.dump({
        "annotations": annotation_collectors,
        "text": comment_pair_map
    }, f)


if __name__ == "__main__":
  main()
