import collections
import glob
import json
import random

INDIVIDUAL_CARD_PAIR_TEMPLATE = """
 <h1 class="subtitle"> %%REVIEW_INFO%% </h1>
 <div class="columns">
    <div class="column">
       <div class="card">
          <div class="card-content">
             <div class="content">
                %%REVIEW_CONTENT%%
             </div>
          </div>
       </div>
    </div>
    <div class="column">
       <div class="card">
          <div class="card-content">
             <div class="content">
                 %%REBUTTAL_CONTENT%%
             </div>
          </div>
       </div>
    </div>
 </div>
"""

HTML_STARTER = """
<HTML>
   <head>
      <title> Review-rebuttal alignment viewer </title>
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.3/css/bulma-rtl.min.css">
   </head>
   <body>
 <div class="container">
"""

HTML_ENDER = """
</div>
   </body>
</HTML>
"""

def highlight_sentences(sentences, highlight_indices):
  html_builder = ""
  for sentence in sentences:
    if sentence["sentence_index"] in highlight_indices:
      html_builder += "<b>" + sentence["text"] + "</b> "
    else:
      html_builder += sentence["text"] + " "
  return html_builder


def get_html_blip(pair, rebuttal_sentence_index):

  review_info = "".join(["<b>Title</b>: ", pair["metadata"]["title"],"<br/>",
  "<b>Rebuttal index</b>: ",  str(rebuttal_sentence_index),  "<br>"])

  html_builder = ""
  selected_rebuttal_sentence = pair["rebuttal_sentences"][rebuttal_sentence_index]
  rebuttal_html = highlight_sentences(pair["rebuttal_sentences"],
  [rebuttal_sentence_index])
  align_type, align_indices = selected_rebuttal_sentence["alignment"]
  review_html =  "".join(["<b>Context type</b>: ", align_type, "<br /> <br/>",])
  if align_type == "context_sentences":
    review_html += highlight_sentences(pair["review_sentences"], align_indices)
  else:
    review_html += highlight_sentences(pair["review_sentences"], [])
  return review_info, review_html, rebuttal_html
  

def get_html_blob(pair):
  blips = []
  for i in range(len(pair["rebuttal_sentences"])):
    review_info, review_html, rebuttal_html = get_html_blip(pair, i)
    blips.append(INDIVIDUAL_CARD_PAIR_TEMPLATE.replace("%%REVIEW_INFO%%", review_info).replace("%%REVIEW_CONTENT%%",
    review_html).replace("%%REBUTTAL_CONTENT%%", rebuttal_html))
  return HTML_STARTER + "\n".join(blips) + HTML_ENDER


def main():

  dataset_dir = "../data_prep/final_dataset/"

  SUBSETS = "train dev test".split()

  datasets = collections.defaultdict(list)

  for subset in SUBSETS:
    for filename in glob.glob(dataset_dir + subset + "/*"):
      with open(filename, 'r') as f:
        datasets[subset].append(json.load(f))

  random.seed(29)

  selected_pairs = random.sample([i for i in  datasets["train"] if
  i["metadata"]["annotator"] ==  'anno10'], 10)

  for pair in selected_pairs:
    html_blob = get_html_blob(pair)
    with open("".join(["html_dumps/",pair["metadata"]["review_id"], ".html"]), 'w')  as f:
      f.write(html_blob.encode('ascii', 'ignore'))


if __name__ == "__main__":
  main()

