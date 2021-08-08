import collections
import json
import tqdm

import logging

import data_prep_lib as dpl

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
        dpl.REVIEW: sentence_map[review_id],
        dpl.REBUTTAL: sentence_map[rebuttal_id],
        dpl.METADATA: dpl.metadata_formatter(fields)
    }
  return comment_pair_map



def collect_annotations():

  with open("final_data_dump/orda_annotations_0516.json", 'r') as f:
    annotations_from_file = json.load(f)

  annotation_collectors = collections.defaultdict(dict)

  for annot in dpl.AnnotationTypes.ALL:
    sorted_rows = sorted(annotations_from_file[annot], key=lambda x: x["pk"])
    for row in sorted_rows:
      rev_id, annotator = dpl.get_key_from_annotation(row)
      if annotator not in annotation_collectors[rev_id]:
        annotation_collectors[rev_id][annotator] = dpl.AnnotationCollector(
            rev_id, annotator)
      annotation_collectors[rev_id][annotator].annotations[annot].append(row)
  return annotation_collectors


def build_filtered_sentence_map(filtered_sentences):
  sentence_map = {}
  for sentence in filtered_sentences:
    all_together_dict = sentence["fields"]
    maybe_labels = dpl.recursive_json_load(all_together_dict["labels"])
    if "0" not in maybe_labels:
      raise dpl.NoLabelError
    else:
      all_together_dict.update(maybe_labels["0"])
    sentence_map[all_together_dict["review_sentence_index"]] = all_together_dict
  return sentence_map

def get_final_review_labels(details):
  asp = details.get('asp', None)
  pol = details.get('pol', None)
  coarse = details.get('arg', None)
  if coarse == 'Structuring':
    fine = "Structuring." + details['struc']
  elif coarse == 'Request':
    fine = "Request." + details['req']
  else:
    fine = None

  return coarse, fine, asp, pol


def build_review_sentences(filtered_sentences, review_text, merge_prev):
  filtered_sentence_map = build_filtered_sentence_map(filtered_sentences)

  post_merge_mapping = []
  
  final_sentence_list = []
  for i, (sentence_text, merge_prev_val) in enumerate(zip(review_text,
    merge_prev)):
    if not merge_prev_val:
      details = filtered_sentence_map[i]
      coarse, fine, asp, pol = get_final_review_labels(details)
      final_sentence_list.append(dpl.ReviewSentence( details["review_id"],
      details["review_sentence_index"], sentence_text, coarse, fine, asp, pol))
      post_merge_mapping.append(details["review_sentence_index"])
    else:
      old = final_sentence_list.pop(-1)
      final_sentence_list.append(dpl.ReviewSentence(
        old.review_id, old.sentence_index, old.text.rstrip() + " " +
        sentence_text.lstrip(), old.coarse, old.fine, old.asp, old.pol
      ))
      post_merge_mapping.append(post_merge_mapping[-1])
  return final_sentence_list, post_merge_mapping


def get_review_sentences(annotation_collection, review_text, merge_prev):
  final_review_sentences = []
  filtered_sentences = dpl.filter_annotations_for_latest(
      annotation_collection, dpl.AnnotationTypes.rev_sent_ann)
  if len(filtered_sentences) == len(review_text) - sum(merge_prev):
    return build_review_sentences(filtered_sentences, review_text, merge_prev)
  else:
    refiltered_sentences = []
    for sentence in filtered_sentences:
      index = sentence["fields"]["review_sentence_index"]
      if merge_prev[index]:
        if dpl.recursive_json_load(sentence["fields"]["labels"]):
          # This should not be empty...
          logging.info("Extra labels in sentence to be merged, index {0}; review_id {1}; annotator {2}".format(index,
          annotation_collection.review_id, annotation_collection.annotator))
      else:
        refiltered_sentences.append(sentence)
    assert len(refiltered_sentences) == len(review_text) - sum(merge_prev)
    return build_review_sentences(refiltered_sentences, review_text, merge_prev)


def get_rebuttal_sentences(annotation_collection, post_merge_map, rebuttal_text):
  final_rebuttal_sentences = []
  review_id = None  # fix
  rebuttal_id = None  # fix
  for sentence in dpl.filter_annotations_for_latest(annotation_collection,
                                                dpl.AnnotationTypes.reb_sent_ann):
    index = sentence["fields"]["rebuttal_sentence_index"]
    aligned_indices = [
        post_merge_map[i]
        for i, val in
        enumerate(json.loads(sentence["fields"]["aligned_review_sentences"]))
        if val
    ]
    final_rebuttal_sentences.append(
        dpl.RebuttalSentence(review_id, rebuttal_id, index, rebuttal_text[index],
                         None, sentence["fields"]["relation_label"],
                         aligned_indices,
                         sentence["fields"]["alignment_category"]))
  return final_rebuttal_sentences


def build_annotation(annotation_collection,
                     text_and_metadata,
                     other_annotators=[]):
  metadata = text_and_metadata[dpl.METADATA]
  metadata["annotator"] = annotation_collection.annotator
  metadata["other_annotators"] = other_annotators

  merge_prev = json.loads(annotation_collection.annotations[
      dpl.AnnotationTypes.rev_ann][0]["fields"]["errors"])["merge_prev"]
  try:
    review_sentences, post_merge_map = get_review_sentences(annotation_collection,
                                               text_and_metadata[dpl.REVIEW],
                                               merge_prev)
  except dpl.NoLabelError:
    logging.info("No label fail here.")
    return None
  rebuttal_sentences = get_rebuttal_sentences(annotation_collection,
                                              post_merge_map,
                                              text_and_metadata[dpl.REBUTTAL])
  return dpl.Annotation(metadata, review_sentences, rebuttal_sentences)


def pick_best_annotation(valid_annotations):
  for best_annotator in dpl.preferred_annotators:
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
  return final_annotation_list, extra_annotation_list


def main():
  comment_pair_map = get_text()
  annotation_collectors = collect_annotations()
  final_annotations, extra_annotations = process_annotations(
    comment_pair_map, annotation_collectors)
  for final_annotation in final_annotations.items():
    print(final_annotation)



if __name__ == "__main__":
  main()
