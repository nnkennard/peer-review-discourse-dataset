"""Convert the outputs form the annotation server into a cleaned dataset."""

import argparse
import collections
import json
import logging
import os
import tqdm

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
parser.add_argument('-a',
                    '--annotation_dump',
                    default="../final_data_dump/orda_annotations_0516.json",
                    type=str,
                    help='path to annotation dump from annotation server')


def get_text(text_dump_path):
  """Get text for all reviews and rebuttals from server dataset dump.

    Args:
      text_dump_path: Path to json dump of text

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
    comment_pair_map[review_id] = {
        dpl.REVIEW: sentence_map[review_id],
        dpl.REBUTTAL: sentence_map[rebuttal_id],
        dpl.METADATA: dpl.metadata_formatter(fields)
    }
  return comment_pair_map


def collect_annotations(annotation_dump_path):
  """Collect all annotations relevant to a particular review-rebuttal pair.
    Annotations are collected for one annotator at a time, and include review
    sentence annotations, rebuttal sentence annotations, and overall review
    annotations.

    Args:
      annotation_dump_path: Path to json dump of annotations

    Returns:
      Map (actually nested dictionary) from (review_id, annotator) to
      data_prep_lib.FrozenAnnotationCollector of relevant annotations.

  """
  with open(annotation_dump_path, 'r') as f:
    annotations_from_file = json.load(f)

  annotation_collectors = collections.defaultdict(dict)

  for annot in dpl.AnnotationTypes.ALL:
    for row in annotations_from_file[annot]:
      rev_id, annotator = dpl.get_key_from_annotation(row)
      # Start a collector if necessary
      if annotator not in annotation_collectors[rev_id]:
        annotation_collectors[rev_id][annotator] = dpl.AnnotationCollector(
            rev_id, annotator)
      annotation_collectors[rev_id][annotator].annotations[annot].append(row)

  frozen_annotation_collectors = collections.defaultdict(dict)
  for k1, v1 in annotation_collectors.items():
    for k2, v2 in v1.items():
      frozen_annotation_collectors[k1][k2] = v2.freeze()

  return frozen_annotation_collectors


def get_review_sentences(annotation_collector, review_text, merge_prev):
  """Convert annotations into final dataset format.
    Args:
      annotation_collector: A FrozenAnnotationCollector
      review_text: list of sentences in the review text
      merge_prev: list of {0,1} values to indicate whether a sentence should be
      merged with the prior sentence.

    Returns:
      final_sentence_list: list of sentences in ReviewSentence format
      post_merge_mapping: mapping from original sentence indices to post-merge
      indices
  """
  final_review_sentences = []
  filtered_sentences = annotation_collector.review_sentences
  if not (len(filtered_sentences) == len(review_text) - sum(merge_prev)):
    refiltered_sentences = []
    for sentence in filtered_sentences:
      index = sentence["fields"]["review_sentence_index"]
      if merge_prev[index]:
        if dpl.recursive_json_load(sentence["fields"]["labels"]):
          # This should not be empty...
          logging.info(
              "{0} {1} Extra labels in sentence to be merged @ {2}"
              .format(annotation_collector.review_id,
                      annotation_collector.annotator, index))
      else:
        refiltered_sentences.append(sentence)
    assert len(refiltered_sentences) == len(review_text) - sum(merge_prev)
    filtered_sentences = refiltered_sentences

  filtered_sentence_map = dpl.build_filtered_sentence_map(filtered_sentences)

  post_merge_mapping = []
  final_sentence_list = []

  for i, (sentence_text,
          merge_prev_val) in enumerate(zip(review_text, merge_prev)):
    if not merge_prev_val:  # Just a normal sentence
      review_sentence_row = filtered_sentence_map[i]
      coarse, fine, asp, pol = dpl.clean_review_label(review_sentence_row)
      final_sentence_list.append(
          dpl.ReviewSentence(review_sentence_row["review_id"],
                             review_sentence_row["review_sentence_index"],
                             sentence_text, coarse, fine, asp, pol))
      post_merge_mapping.append(review_sentence_row["review_sentence_index"])
    else:
      # Should merge with previous sentence. This sentence should have no-op
      # labels, so we can use the labels from the previous sentence and just
      # tack on the text from the current sentence to the old sentence.

      # TODO: add an assert here to make sure there is no required stuff in the
      # label

      old = final_sentence_list.pop(-1)
      merged_sentence = old.text.rstrip() + " " + sentence_text.lstrip()
      final_sentence_list.append(
          dpl.ReviewSentence(old.review_id, old.sentence_index, merged_sentence,
                             old.coarse, old.fine, old.asp, old.pol))
      post_merge_mapping.append(post_merge_mapping[-1])

  return final_sentence_list, post_merge_mapping


def get_rebuttal_sentences(annotation_collector, post_merge_map, rebuttal_text):
  final_rebuttal_sentences = []

  (review_id, rebuttal_id), = set([
      (dpl.get_fields(i)["review_id"], dpl.get_fields(i)["rebuttal_id"])
      for i in annotation_collector.rebuttal_sentences
  ])

  for sentence in annotation_collector.rebuttal_sentences:
    index, label, coarse, alignment = dpl.clean_rebuttal_label(
        sentence, post_merge_map)

    final_rebuttal_sentences.append(
        dpl.RebuttalSentence(review_id, rebuttal_id, index,
                             rebuttal_text[index], coarse, label, alignment))
  return final_rebuttal_sentences


def build_annotation(annotation_collector,
                     text_and_metadata,
                     other_annotators=[]):
  print("I'm seeing the annotator ", annotation_collector.annotator)
  metadata = text_and_metadata[dpl.METADATA]
  metadata["annotator"] = annotation_collector.annotator
  metadata["other_annotators"] = other_annotators

  merge_prev = json.loads(
      dpl.get_fields(
          annotation_collector.review_annotation)["errors"])["merge_prev"]

  try:
    review_sentences, post_merge_map = get_review_sentences(
        annotation_collector, text_and_metadata[dpl.REVIEW], merge_prev)
  except dpl.NoLabelError:
    logging.info("No label fail here.")
    return None
  rebuttal_sentences = get_rebuttal_sentences(annotation_collector,
                                              post_merge_map,
                                              text_and_metadata[dpl.REBUTTAL])
  print("At the end, annotator:", metadata["annotator"])
  return dpl.Annotation(review_sentences, rebuttal_sentences, metadata)


def order_annotations_by_preference(collector_dict):
  valid_annotations = {
      collector.annotator: collector
      for collector in collector_dict.values()
      if collector is not None
  }
  ordered_annotators = []
  for best_annotator in dpl.preferred_annotators:
    if best_annotator in valid_annotations:
      ordered_annotators.append(valid_annotations[best_annotator])
      print("&", best_annotator)
  return ordered_annotators


def process_annotations(comment_pair_map, annotation_collectors):
  extra_annotation_list = []
  final_annotation_list = []

  for review_id, collector_dict in tqdm.tqdm(annotation_collectors.items()):
    print("*"* 80)
    print(review_id)
    if review_id == "example_review":
      continue
    ordered_annotations = order_annotations_by_preference(collector_dict)
    if not ordered_annotations:
      continue
    best_annotation = ordered_annotations.pop(0)
    final_annotation_list.append(build_annotation(best_annotation,
    comment_pair_map[review_id]))
    for other_annotation in ordered_annotations:
      extra_annotation_list.append(build_annotation(other_annotation,
    comment_pair_map[review_id]))

    if len(final_annotation_list) > 2:
      break
  
  return final_annotation_list, extra_annotation_list


def write_annotations_to_dir(annotations, dir_name, append_annotator=False):
  for subdir in ["", "train", "dev", "test"]:
    subdir_name = dir_name + "/" + subdir
    if not os.path.exists(subdir_name):
      os.makedirs(subdir_name)

  for annotation in annotations:
    if annotation is None:
      continue
    annotation.write_to_dir(dir_name, append_annotator)


def main():
  args = parser.parse_args()
  comment_pair_map = get_text(args.text_dump)
  annotation_collectors = collect_annotations(args.annotation_dump)
  final_annotations, extra_annotations = process_annotations(
      comment_pair_map, annotation_collectors)
  write_annotations_to_dir(final_annotations, "final_dataset/")
  write_annotations_to_dir(extra_annotations,
                           "extra_annotations/",
                           append_annotator=True)


if __name__ == "__main__":
  main()
