import collections
import json
import logging

logging.basicConfig(filename="log_prep_data.log")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Constants for later

REVIEW_ID, INITIALS = "review_id initials".split()
REVIEW, REBUTTAL, METADATA = "review rebuttal metadata".split()
META_FIELDS_TO_EXTRACT = 'forum_id review_id rebuttal_id title reviewer'.split()
PERMALINK_FORMAT = "https://openreview.net/forum?id={0}&noteId={1}"

class AnnotationTypes(object):
  rev_ann = "reviewannotation"
  rev_sent_ann = "reviewsentenceannotation"
  reb_sent_ann = "rebuttalsentenceannotation"
  ALL = [rev_ann, rev_sent_ann, reb_sent_ann]

ReviewSentence = collections.namedtuple(
    "ReviewSentence",
    "review_id sentence_index text coarse fine asp pol".split())
RebuttalSentence = collections.namedtuple(
    "RebuttalSentence",
    "review_id rebuttal_id sentence_index text coarse fine alignment alignment_type"
    .split())

class Annotation(object):

  def __init__(self, review_sentences, rebuttal_sentences, meta, is_main=True):
    self.meta = meta
    self.review_sentences = review_sentences
    self.rebuttal_sentences = rebuttal_sentences
    self.is_main = is_main

  def write_to_dir(self, dir_name):
    name_builder = self.meta["review_id"]
    if not self.is_main:
      name_builder += "." + self.meta["annotator"]

    filename = "".join([dir_name, "/", name_builder, ".json"])
    with open(filename, 'w') as f:
      f.write(self.__repr__())

  def __repr__(self):
    return json.dumps({
        "metadata":
            self.meta,
        "review_sentences": [sent._asdict() for sent in self.review_sentences],
        "rebuttal_sentences": [
            sent._asdict() for sent in self.rebuttal_sentences
        ]
    })

class AnnotationCollector(object):

  def __init__(self, review_id, annotator):
    self.review_id = review_id
    self.annotator = annotator
    self.annotations = {
        annotation_type: [] for annotation_type in AnnotationTypes.ALL
    }

  def is_valid(self):
    if self.review_id == 'example_review':
      logging.info("Skipping example review annotated by {0}".format(
          self.annotator))
      return False
    for annotation_type, annotation_list in self.annotations.items():
      if not annotation_list:
        logging.info("{0} {1} missing {2}".format(self.review_id,
                                                  self.annotator,
                                                  annotation_type))
        return False
    return True

  def __repr__(self):
    return "\n".join([
        "Annotation collector for {0} annotated by {1}".format(
            self.review_id, self.annotator)
    ] + [
        "{0} : {1}".format(ann_type, len(anns))
        for ann_type, anns in self.annotations.items()
    ] + [""])


with open('annomap.json', 'r') as f:
  ANONYMIZER = json.load(f)

TYPE_TO_KEY_MAP = {
    AnnotationTypes.rev_sent_ann: "review_sentence_index",
    AnnotationTypes.reb_sent_ann: "rebuttal_sentence_index",
}

preferred_annotators = ["anno{0}".format(i) for i in range(20)]  # Fix this!!


# Utilities

def get_key_from_annotation(ann):
  return ann["fields"][REVIEW_ID], ANONYMIZER[ann["fields"][INITIALS]]

def metadata_formatter(fields):
  metadata = {k: v for k, v in fields.items() if k in META_FIELDS_TO_EXTRACT}
  if fields["dataset"].startswith("traindev"):
    metadata["conference"] = "ICLR2019"
  else:
    metadata["conference"] = "ICLR2020"
  metadata["permalink"] = PERMALINK_FORMAT.format(
          metadata["forum_id"], metadata["rebuttal_id"])
  return metadata

class NoLabelError(Exception):
  pass

def recursive_json_load(in_repr):
  while type(in_repr) == str:
    in_repr = json.loads(in_repr)
  return in_repr

def clean_review_sentence_dict(rev_sent_dict):
  new_map = {}
  for k, v in rev_sent_dict["fields"].items():
    if k == "labels":
      labels = recursive_json_load(v)
      if not len(labels):
        raise NoLabelError
      new_map.update(labels["0"])
    else:
      new_map[k] = v
  return new_map


def filter_annotations_for_latest(annotation_collection, annotation_type):
  final_annotations = {}
  key = TYPE_TO_KEY_MAP[annotation_type]
  for annotation in sorted(annotation_collection.annotations[annotation_type],
                           key=lambda x: x["pk"]):
    final_annotations[annotation["fields"][key]] = annotation
  final_list = [final_annotations[k] for k in sorted(final_annotations.keys())]
  return final_list


