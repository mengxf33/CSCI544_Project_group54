import networkx as nx
from itertools import combinations
from collections import Counter
from preprocess import *
from basic import basic_recommender

G = nx.Graph()
filename = '../processed_data/processed_train.csv'
# filename = '../tiny_train.csv'
for line in line_stream(filename):
    ID, title, body, tags = line
    tags = tags.split()
    # add nodes
    for tag in tags:
        if tag:
            if not G.has_node(tag):
                G.add_node(tag)
                G.node[tag]['tag_count'] = 1
            else:
                G.node[tag]['tag_count'] += 1
    # add tag->tag edges
    for edge in combinations(tags, 2):
        ni, nj = edge
        if not G.has_edge(ni, nj):
            G.add_edge(ni, nj, weight=1)
        else:
            G.edge[ni][nj]['weight'] += 1

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

def tag_to_tag_recommender(tags, n_recs):

    """Given a Counter with 'tags: scores` generate and score associated tags"""

    total_scores = Counter()
    # eliminate tokens with 0 usefulness(not in the training set)
    tags -= Counter()
    for tag in tags:
        tag_scores = Counter({nj: tags[tag] * G.edge[tag][nj]['weight'] / G.node[tag]['tag_count']
                              for _, nj in G.edges(tag)})
        tag_scores = Counter(dict(tag_scores.most_common(n_recs)))  # keep best n_recs
        total_scores = combine_tags(total_scores, tag_scores)
    return total_scores

# title = "Creating a repetitive node jquery html from a hash array with simplexml_load_string, a cycle and variables"
# tags = basic_recommender(title)
# print(tag_to_tag_recommender(tags, 3))