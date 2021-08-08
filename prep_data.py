# Imports

import collections
import json
import logging
import tqdm

logging.basicConfig(filename="example2.log")
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

  def __init__(self, review_sentences, rebuttal_sentences, meta):
    self.meta = meta
    self.review_sentences = review_sentences
    self.rebuttal_sentences = rebuttal_sentences

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


def clean_review_sentence_dict(rev_sent_dict):
  new_map = {}
  for k, v in rev_sent_dict["fields"].items():
    if k == "labels":
      labels = json.loads(json.loads(v))
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


def get_text():
  with open("final_data_dump/orda_text_0415.json", 'r') as f:
    j = json.load(f)

  sentence_map = collections.defaultdict(list)
  for sentence in j["sentence"]:
    fields = sentence["fields"]
    assert fields["sentence_index"] == len(sentence_map[fields["comment_id"]])
    sentence_map[fields["comment_id"]].append(fields["text"])

  comment_pair_map = {}
  for example in j["example"]:
    fields = example["fields"]
    review_id, rebuttal_id = fields["review_id"], fields["rebuttal_id"]
    comment_pair_map[review_id] = {
        REVIEW: sentence_map[review_id],
        REBUTTAL: sentence_map[rebuttal_id],
        METADATA: metadata_formatter(fields)
    }
  return comment_pair_map



def collect_annotations():

  with open("final_data_dump/orda_annotations_0516.json", 'r') as f:
    annotations_from_file = json.load(f)

  annotation_collectors = collections.defaultdict(dict)

  for annot in AnnotationTypes.ALL:
    sorted_rows = sorted(annotations_from_file[annot], key=lambda x: x["pk"])
    for row in sorted_rows:
      rev_id, annotator = get_key_from_annotation(row)
      if annotator not in annotation_collectors[rev_id]:
        annotation_collectors[rev_id][annotator] = AnnotationCollector(
            rev_id, annotator)
      annotation_collectors[rev_id][annotator].annotations[annot].append(row)
  return annotation_collectors


def get_review_sentences(annotation_collection, review_text, merge_prev):
  final_review_sentences = []
  filtered_sentences = filter_annotations_for_latest(
      annotation_collection, AnnotationTypes.rev_sent_ann)
  review_id = None
  print(len(merge_prev) - len(filtered_sentences))
  for sentence in filtered_sentences:
    index = sentence["fields"]["review_sentence_index"]
    labels = clean_review_sentence_dict(sentence)
    print(review_text[index])
    print(labels)
  print()
  return None, None


def get_rebuttal_sentences(annotation_collection, rebuttal_text):
  final_rebuttal_sentences = []
  review_id = None  # fix
  rebuttal_id = None  # fix
  for sentence in filter_annotations_for_latest(annotation_collection,
                                                AnnotationTypes.reb_sent_ann):
    index = sentence["fields"]["rebuttal_sentence_index"]
    aligned_indices = [
        i
        for i, val in enumerate(sentence["fields"]["aligned_review_sentences"])
        if val
    ]
    final_rebuttal_sentences.append(
        RebuttalSentence(review_id, rebuttal_id, index, rebuttal_text[index],
                         None, sentence["fields"]["relation_label"],
                         aligned_indices,
                         sentence["fields"]["alignment_category"]))
  return final_rebuttal_sentences


def build_annotation(annotation_collection,
                     text_and_metadata,
                     other_annotators=[]):
  metadata = text_and_metadata[METADATA]
  metadata["annotator"] = annotation_collection.annotator
  metadata["other_annotators"] = other_annotators

  merge_prev = json.loads(annotation_collection.annotations[
      AnnotationTypes.rev_ann][0]["fields"]["errors"])["merge_prev"]
  review_sentences, _ = get_review_sentences(annotation_collection,
                                             text_and_metadata[REVIEW],
                                             merge_prev)
  rebuttal_sentences = get_rebuttal_sentences(annotation_collection,
                                              text_and_metadata[REBUTTAL])
  return Annotation(metadata, review_sentences, rebuttal_sentences)


def pick_best_annotation(valid_annotations):
  for best_annotator in preferred_annotators:
    if best_annotator in valid_annotations:
      return (valid_annotations[best_annotator], [
          annotation for annotator, annotation in valid_annotations.items()
          if not annotator == best_annotator
      ])
  assert False  # Iterating through all possible annotators, we should never reach here.

def process_annotations(comment_pair_map, annotation_collectors):
  extra_annotation_list = []
  final_annotation_list = []

  for review_id, collector_dict in tqdm.tqdm(annotation_collectors.items()):
    valid_annotations = {
        collector.annotator: collector
        for collector in collector_dict.values()
        if collector.is_valid()
    }
    if valid_annotations:
      main_annotation, other_annotations = pick_best_annotation(valid_annotations)
      final_annotation_list.append(
          build_annotation(main_annotation, comment_pair_map[review_id],
                           [oa.annotator for oa in other_annotations]))
      extra_annotation_list += [
          build_annotation(extra_annotation, comment_pair_map[review_id])
          for extra_annotation in other_annotations
      ]

def main():
  comment_pair_map = get_text()
  annotation_collectors = collect_annotations()
  filter_annotations(annotation_collectors)
  process_annotations(comment_pair_map, annotation_collectors)


if __name__ == "__main__":
  main()
