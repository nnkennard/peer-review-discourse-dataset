import argparse
import json
import os

import data_prep_lib as dpl

parser = argparse.ArgumentParser(
    description="Convert the filtered database entries into a cleaned dataset.")
parser.add_argument('-i',
                    '--intermediate_file',
                    default="filtered_database.json",
                    type=str,
                    help='path to text dump from annotation server')


def read_filtered_dataset(intermediate_file):
  with open(intermediate_file, 'r') as f:
    obj = json.load(f)
    return obj["annotations"], obj["text"]

def process_review_sentences(review_sentence_annotations, review_text,
  merge_prev):

  if not (len(review_sentence_annotations)
          == len(review_text) - sum(merge_prev)):
    filtered_sentences = {}
    for str_index in sorted(review_sentence_annotations.keys()):
      sentence = review_sentence_annotations[str_index]
      if merge_prev[int(str_index)]:
        # TODO: W are not keeping this! make sure we get the text elsewhere
        if dpl.recursive_json_load(sentence["labels"]):
          # This is expected to be empty
          # TODO: Log this exception
          return None, None
      else:
        filtered_sentences[str_index] = sentence
    review_sentence_annotations = filtered_sentences

  assert (len(review_sentence_annotations)
          == len(review_text) - sum(merge_prev))

  original_index_to_merged_index = []
  final_sentence_list = []

  for i, (sentence_text, merge_prev_val) in enumerate(zip(review_text,
  merge_prev)):
    if merge_prev_val:
      assert i > 0
      # Need to merge just the text with previous sentence
      old = final_sentence_list.pop(-1)
      merged_text = old.text.rstrip() + " " + sentence_text.lstrip()
      final_sentence_list.append(
          dpl.ReviewSentence(old.review_id, old.sentence_index, merged_text,
                             old.coarse, old.fine, old.asp, old.pol))
      original_index_to_merged_index.append(original_index_to_merged_index[-1])
    else:
      relevant = review_sentence_annotations[str(i)]
      assert i == relevant["review_sentence_index"]
      coarse, fine, asp, pol = dpl.clean_review_label(relevant)
      final_sentence_list.append(
          dpl.ReviewSentence(relevant["review_id"],
                             #review_sentence_row["review_sentence_index"],
                             i,
                             sentence_text, coarse, fine, asp, pol))
      original_index_to_merged_index.append(relevant["review_sentence_index"])


  return final_sentence_list, original_index_to_merged_index


def process_rebuttal_sentences(rebuttal_sentence_annotations, rebuttal_text, merge_map):
  final_rebuttal_sentences = []

  (review_id, rebuttal_id), = set([
      (dpl.get_fields(i)["review_id"], dpl.get_fields(i)["rebuttal_id"])
      for i in rebuttal_sentence_annotations
  ])

  for sentence in rebuttal_sentence_annotations:
    index, label, coarse, alignment = dpl.clean_rebuttal_label(
        sentence, merge_map)

    final_rebuttal_sentences.append(
        dpl.RebuttalSentence(review_id, rebuttal_id, index,
                             rebuttal_text[index], coarse, label, alignment))
  return final_rebuttal_sentences




def process_annotation(annotation, text):
  metadata = dict(text[dpl.METADATA])
  metadata["annotator"] = annotation["annotator"]
  merge_prev = json.loads(
      dpl.get_fields(annotation["review_annotation"])["errors"])["merge_prev"]

  processed_review, merge_map = process_review_sentences(
      annotation["review_sentences"],text["review"], merge_prev)
  if processed_review is not None:
    processed_rebuttal = process_rebuttal_sentences(
    annotation["rebuttal_sentences"], text["rebuttal"], merge_map)
    return dpl.Annotation(processed_review, processed_rebuttal, metadata)
  else:
    return None


def process_all_annotations(annotation_collections, text_map):
  final_annotations = []
  extra_annotations = []
  for review_id, annotations in annotation_collections.items():
    if review_id == "example_id":
      continue
    annotators = sorted(annotations.keys(),
                        key=lambda x: dpl.preferred_annotators.index(x))
    assert annotators[0] == "anno0" or "anno0" not in annotators
    maybe_valid_annotations = [
        process_annotation(annotations[annotator], text_map[review_id])
        for annotator in annotators
        if annotations[annotator] is not None
    ]
    valid_annotations = [
        annotation for annotation in maybe_valid_annotations if annotation is not None
    ]
    if valid_annotations:
      final_annotations.append(valid_annotations.pop(0))
      extra_annotations += valid_annotations
  return final_annotations, extra_annotations

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

  annotation_collections, text_map = read_filtered_dataset(
      args.intermediate_file)

  final_annotations, extra_annotations = process_all_annotations(
      annotation_collections, text_map)
  write_annotations_to_dir(final_annotations, "final_dataset/")
  write_annotations_to_dir(extra_annotations,
                             "extra_annotations/",
                             append_annotator=True)




if __name__ == "__main__":
  main()
