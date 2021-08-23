import collections
import json
import logging
import yaml

logging.basicConfig(filename="log_prep_data.log")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Constants for later

REVIEW, REBUTTAL, METADATA = "review rebuttal metadata".split()
META_FIELDS_TO_EXTRACT = 'forum_id review_id rebuttal_id title reviewer'.split()
PERMALINK_FORMAT = "https://openreview.net/forum?id={0}&noteId={1}"


class ServerModels(object):
  example = "example"
  sentence = "sentence"


class AnnotationFields(object):
  sentence_index = "sentence_index"
  comment_id = "comment_id"
  text = "text"
  review_id = "review_id"
  rebuttal_id = "rebuttal_id"


ReviewSentence = collections.namedtuple(
    "ReviewSentence",
    "review_id sentence_index text coarse fine asp pol".split())
RebuttalSentence = collections.namedtuple(
    "RebuttalSentence",
    "review_id rebuttal_id sentence_index text coarse fine alignment".split())

with open("label_map.yaml") as stream:
  map_of_maps = yaml.safe_load(stream)
  LABEL_MAP, REBUTTAL_FINE_TO_COARSE, CONTEXT_TYPE_MAP = [
      map_of_maps[key] for key in
      "label_name_map rebuttal_fine_to_coarse rebuttal_alignment_type".split()
  ]


class Annotation(object):

  def __init__(self, review_sentences, rebuttal_sentences, meta):
    self.meta = meta
    self.review_sentences = review_sentences
    self.rebuttal_sentences = rebuttal_sentences

  def write_to_dir(self, dir_name, add_annotator_name=False):
    try:
      name_builder = "".join(
          [SUBSET_MAP[self.meta["forum_id"]], "/", self.meta["review_id"]])
    except KeyError:
      logging.info("{0}\tForum not in train-test split".format(
          self.meta["review_id"]))
      return
    if add_annotator_name:
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


with open('subset_map.json', 'r') as f:
  SUBSET_MAP = json.load(f)

preferred_annotators = ["anno{0}".format(i) for i in range(20)
                       ]  # TODO: Fix this!!
                       

# Utilities


def metadata_formatter(fields):
  metadata = {k: v for k, v in fields.items() if k in META_FIELDS_TO_EXTRACT}
  if fields["dataset"].startswith("traindev"):
    metadata["conference"] = "ICLR2019"
  else:
    metadata["conference"] = "ICLR2020"
  metadata["permalink"] = PERMALINK_FORMAT.format(metadata["forum_id"],
                                                  metadata["rebuttal_id"])
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


def clean_review_label(review_sentence_row):
  """Convert review labels to uniform format."""
  asp = review_sentence_row.get('asp', None)
  pol = review_sentence_row.get('pol', None)
  coarse = review_sentence_row.get('arg', None)
  if coarse == 'Structuring':
    fine = "Structuring." + review_sentence_row['struc']
  elif coarse == 'Request':
    fine = "Request." + review_sentence_row['req']
  else:
    fine = None
  return (LABEL_MAP[label] for label in [coarse, fine, asp, pol])


Alignment = collections.namedtuple("Alignment",
                                   "category aligned_indices".split())


def clean_rebuttal_label(rebuttal_sentence_row, merge_map):
  index, aligned_array, raw_label, raw_category = [
      get_fields(rebuttal_sentence_row)[key] for key in [
          "rebuttal_sentence_index", "aligned_review_sentences",
          "relation_label", "alignment_category"
      ]
  ]

  label = LABEL_MAP[raw_label]
  coarse = REBUTTAL_FINE_TO_COARSE[label]

  aligned_indices = [
      merge_map[i] for i, val in enumerate(json.loads(aligned_array)) if val
  ]

  if aligned_indices == list(range(len(aligned_indices))):
    alignment = Alignment("context_global", None)
  elif not aligned_indices:
    assert False
  else:
    assert aligned_indices
    category = CONTEXT_TYPE_MAP[raw_category]
    if category == "context_sentences":
      alignment = Alignment(category, aligned_indices)
    else:
      alignment = Alignment(category, None)

  return index, label, coarse, alignment


def get_fields(dataset_row):
  return dataset_row["fields"]


def build_filtered_sentence_map(filtered_sentences):
  """Remap sentence annotations to map from index

    TODO: Fold this into filtering once filtering is added to collection?
  """
  sentence_map = {}
  for sentence in filtered_sentences:
    all_together_dict = sentence["fields"]
    maybe_labels = recursive_json_load(all_together_dict["labels"])
    if "0" not in maybe_labels:
      raise NoLabelError
    else:
      all_together_dict.update(maybe_labels["0"])
    sentence_map[all_together_dict["review_sentence_index"]] = all_together_dict
  return sentence_map
