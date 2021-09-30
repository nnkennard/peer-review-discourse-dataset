import collections
import glob
import json
import sys
import yaml

TABLE_START_OPEN = r"""
\begin{table*}[]
\begin{tabular}{"""
TABLE_START_CLOSE = r"""}
\toprule
"""
TABLE_END = r"""
\bottomrule
\end{tabular}
\end{table*}
"""

def rebuttal_description_table(info, category_namesi, overall_counts):
  rows = collections.defaultdict(list)
  arg_names = {"-":"-"}
  for label, label_info in info.items():
    if label_info["domain"] == "review":
      if label_info["category"] == "review_function":
        arg_names[label] = label_info["nice_name"]
      continue
    rows[label_info["category"]].append(
        (label, label_info["nice_name"], label_info["description"],
        label_info["reply-to"]))

  relevant_counts = {}
  for row in sum(rows.values(), []):
    relevant_counts[row[0]] = overall_counts[row[0]]
  percentages = {}
  for k, v in relevant_counts.items():
    percentages[k] = v/sum(relevant_counts.values())

  table_string = "Category & & Label & Description & Reply to & Percentage \\\\ \n \\midrule \n"

  for category, category_rows in rows.items():
    this_category_num_rows = len(category_rows)
    for row in sorted(category_rows, key=lambda x:percentages[x[0]], reverse=True):
      table_string += "".join(
            ["& & ", row[1], r" & ", row[2], " & ", arg_names[row[3]], " & ",
            "{:.2%}".format(percentages[row[0]]).replace("%", "\\%"),  r" \\", '\n'])

  return build_booktabs_table(table_string, 6)

def review_description_table(info, category_names, overall_counts,
  num_review_sentences):
  rows = collections.defaultdict(list)
  for label, label_info in info.items():
    if label in ["arg_other", "arg-request_result"]:
      print(label)
      continue
    if label_info["domain"] == "rebuttal":
      continue
    rows[label_info["category"]].append(
        (label_info["nice_name"], label_info["description"],
        overall_counts[label]
        ))

  table_string = ""

  for category, category_rows in rows.items():
   for row in sorted(category_rows, reverse=True, key=lambda x:x[-1]):
      table_string += "".join(
          [" & & ", row[0], r" & ", row[1], " & ",
          "{:.2%}".format(row[2]/num_review_sentences).replace("%", "\\%"),
          r" \\", '\n'])
    
  return build_booktabs_table(table_string, 5)


def build_booktabs_table(table_content, num_cols):
  return "\n".join([TABLE_START_OPEN + "l" * num_cols + TABLE_START_CLOSE, table_content, TABLE_END])


def main():

  dataset_dir = "../data_prep/final_dataset/"

  SUBSETS = "train dev test".split()

  datasets = collections.defaultdict(list)

  for subset in SUBSETS:
      for filename in glob.glob(dataset_dir + subset + "/*"):
          with open(filename, 'r') as f:
              datasets[subset].append(json.load(f))

  all_pairs = sum(datasets.values(), [])

  overall_counts = collections.Counter()
  num_review_sentences = 0
  for pair in all_pairs:
    num_review_sentences += len(pair["review_sentences"])
    for sentence in pair["review_sentences"]:
      for key in "coarse fine pol asp".split():
        overall_counts[sentence[key]] += 1
    for sentence in pair["rebuttal_sentences"]:
      for key in "coarse fine".split():
        overall_counts[sentence[key]] += 1


  with open("manuscript-lookups.yaml", 'r') as f:
    k = yaml.safe_load(f)
    p = review_description_table(k["label_descriptions"], k["category_names"],
    overall_counts, num_review_sentences)
    print(p)

    print()

    p = rebuttal_description_table(k["label_descriptions"],
    k["category_names"], overall_counts)
    print(p)


if __name__ == "__main__":
  main()
