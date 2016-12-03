from collections import Counter
from basic import basic_recommender
from tag_to_tag import tag_to_tag_recommender
from title_to_tag import title_to_tag_recommender
from preprocess import *

def combine_tags(c1, c2):

    """Probabilistic union of two tag sets.
    c1, c2: Counters with tags as keys, probs as values
    Example:
        c1 = Counter({'php': 0.2, 'xml': 0.14})
        c2 = Counter({'php': 0.1, 'jquery': 0.1})
        combine_tags(c1,c2)"""

    for w in c1:
        if w in c2:
           c1[w] = 1 - (1 - c1[w]) * (1 - c2[w])
    return Counter(dict(c2, **c1)) # add missing items from c2

def normalize(rec, weight=1):
    """Normalizes each recommendation proportionally to the score of the leading tag"""
    try:
        max_score = rec[max(rec, key=rec.get)]
        for tag in rec:
            rec[tag] = weight * rec[tag] / max_score
    except:
        rec = Counter() # empty recommendation
    return rec

def meta_recommender(recomendations, weights):
    """Probabilistically combine `recomendations` with corresponding `weights`"""
    total = Counter()
    for rec, weight in zip(recomendations, weights):
        rec = normalize(rec, weight)
        total = combine_tags(total, rec)    # accumulate tag scores
    return total

def select_tags(rec, threshold=0.1):
    """Select at most 3 tags with score greater than `threshold`"""
    return [tag for tag, score in rec.most_common(3) if score > threshold]

# calculate precision, recall and F1 score
tp = 0
fp = 0
fn = 0
filename = '../processed_data/processed_test.csv'
# filename = '../tiny_test.csv'
for line in line_stream(filename):
    ID, title, body, tags = line
    basic = basic_recommender(title)
    tag_to_tag = tag_to_tag_recommender(basic, 10)
    title_to_tag = title_to_tag_recommender(basic, 10)
    recomendations = meta_recommender([basic, tag_to_tag, title_to_tag], [0.8, 0.2, 0.2])
    predicted_tags = set(select_tags(recomendations))
    true_tags = set(tags.split())
    for tag in predicted_tags:
        if tag not in true_tags:
            fp += 1
        else:
            tp += 1
    for tag in true_tags:
        if tag not in predicted_tags:
            fn += 1

precision = tp / (tp + fp)
recall = tp / (tp + fn)
print('precision is', precision)
print('recall is', recall)
print('F1 score is', 2 * precision * recall / (precision + recall))

# title = "How to fetch an XML feed using asp.net <p>I've decided to convert a Windows Phone 7 app that fetches an XML feed and then parses it to an asp.net web app, using Visual Web Developer Express."
# basic = basic_recommender(title)
# tag_to_tag = tag_to_tag_recommender(basic, 10)
# title_to_tag = title_to_tag_recommender(basic, 10)
# recomendations = meta_recommender([basic, tag_to_tag, title_to_tag], [0.60, 0.33, 0.33])
# print(select_tags(recomendations))

