import collections
import sys
import yaml

TABLE_START_OPEN = r"""
\begin{table*}[]
\begin{tabular}{"""
TABLE_START_CLOSE = r"""}
\toprule
"""
TABLE_END = r"""
\\ \bottomrule
\end{tabular}
\end{table*}
"""

def rebuttal_description_table(info, category_names):
  rows = collections.defaultdict(list)
  arg_names = {"-":"-"}
  for label, label_info in info.items():
    if label_info["domain"] == "review":
      if label_info["category"] == "review_function":
        arg_names[label] = label_info["nice_name"]
      continue
    rows[label_info["category"]].append(
        (label_info["nice_name"], label_info["reply-to"]))

  print(arg_names)

  for k, v in rows.items():
    print(k)
    for i in v:
      print(i)

  table_string = ""

  for category, category_rows in rows.items():
    this_category_num_rows = len(category_rows)
    first_row = category_rows.pop(0)
    table_string += "".join([
          r"\multicolumn{2}{c}{\multirow{",
          str(this_category_num_rows), r"}{*}{\rotatebox[origin=c]{90}{",
          category_names[category], r"}}} & ", first_row[0], r" & ",
          arg_names[first_row[1]],
          r" \\", '\n'
      ])
    for row in category_rows:
      table_string += "".join(
            ["& ", row[0], r" & ", arg_names[row[1]], r" \\", '\n'])

  return build_booktabs_table(table_string, 5)

def review_description_table(info, category_names):
  rows = collections.defaultdict(list)
  for label, label_info in info.items():
    if label_info["domain"] == "rebuttal":
      continue
    rows[label_info["category"]].append(
        (label_info["nice_name"], label_info["description"]))

  table_string = ""
  sub_started = False

  for category, category_rows in rows.items():
    this_category_num_rows = len(category_rows)
    first_row = category_rows.pop(0)
    if 'sub' not in category:
      table_string += "".join([
          r"\multicolumn{2}{c}{\multirow{",
          str(this_category_num_rows), r"}{*}{\rotatebox[origin=c]{90}{",
          category_names[category], r"}}} & ", first_row[0], r" & ", first_row[1],
          r" \\", '\n'
      ])
      for row in category_rows:
        table_string += "".join(
            [" & & ", row[0], r" & ", row[1], r" \\", '\n'])
    else:
      if sub_started:
        table_string += "".join([
          r" & \multirow{",
          str(this_category_num_rows), r"}{*}{\rotatebox[origin=c]{90}{",
          category_names[category], r"}} & ", first_row[0], r" & ", first_row[1],
          r" \\", '\n'
      ])
      else:
         table_string += "".join([
          r"\multirow{8}{*}{\rotatebox[origin=c]{90}{Subtypes}} & ",
          r"\multirow{",
          str(this_category_num_rows), r"}{*}{\rotatebox[origin=c]{90}{",
          category_names[category], r"}} & ", first_row[0], r" & ", first_row[1],
          r" \\", '\n'
      ])
         sub_started = True
      for row in category_rows:
        table_string += "".join(
            [" & & ", row[0], r" & ", row[1], r" \\", '\n'])

    if category == "sub_request":
      break
    if sub_started:
      table_string += r"\cmidrule(lr){2-4}" + "\n"
    else:
      table_string += r"\cmidrule(lr){1-4}" + "\n"


  return build_booktabs_table(table_string, 4)


def build_booktabs_table(table_content, num_cols):
  return "\n".join([TABLE_START_OPEN + "l" * num_cols + TABLE_START_CLOSE, table_content, TABLE_END])


def main():

  with open("manuscript-lookups.yaml", 'r') as f:
    k = yaml.safe_load(f)
    p = review_description_table(k["label_descriptions"], k["category_names"])
    p = rebuttal_description_table(k["label_descriptions"], k["category_names"])
    print(p)


if __name__ == "__main__":
  main()
