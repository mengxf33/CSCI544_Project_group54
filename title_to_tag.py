import networkx as nx
from collections import Counter
from preprocess import *
from basic import basic_recommender

Y = nx.Graph()
filename = '../processed_data/processed_train.csv'
# filename = '../tiny_train.csv'
for line in line_stream(filename):
    ID, title, body, tags = line
    tags = tags.split()
    title = title_tokenize(title)
    for word in title:
        # add word nodes
        if word and not Y.has_node(word):
            Y.add_node(word, is_word=1)
            Y.node[word]['count'] = 1
        elif Y.node[word].get('is_word') == 1:
            Y.node[word]['count'] += 1
        else:
            Y.node[word]['is_word'] = 1
            Y.node[word]['count'] = 1
        # add tag nodes
        for tag in tags:
            if tag and not Y.has_node(tag):
                Y.add_node(tag, is_tag=1)
            else:
                Y.node[tag]['is_tag'] = 1
            # add word->tag edges
            if not Y.has_edge(word, tag):
                Y.add_edge(word, tag, weight=1)
            else:
                Y.edge[word][tag]['weight'] += 1

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

def title_to_tag_recommender(tags, n_recs):
    total_scores = Counter()
    for word in tags:
        if Y.has_node(word) and Y.node[word].get('count'):
            tag_scores = Counter({nj: Y.edge[word][nj]['weight'] / Y.node[word]['count']
                                  for _,nj in Y.edges(word)
                                  if Y.node[nj].get('is_tag') == 1})
            tag_scores = Counter(dict(tag_scores.most_common(n_recs)))
            total_scores = combine_tags(total_scores, tag_scores)
    return total_scores - Counter()

# title = "Creating a repetitive node from a hash array with simplexml_load_string, a cycle and variables"
# tags = basic_recommender(title)
# print(title2tagrecommender(tags, 3))