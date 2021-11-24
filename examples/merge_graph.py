import argparse
import json
from collections import defaultdict
from tqdm import tqdm 
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument('--graphs', nargs='+',
                    help='list of input json graphs')
parser.add_argument('--weights', type=int, nargs='+',
                    help='list of corresponding weights')
parser.add_argument('--output',
                    help='out')
args = parser.parse_args()

assert type(args.graphs) ==list and type(args.weights) ==list, "Graphs and weights must not be empty"
assert len(args.graphs) == len(args.weights), "Graphs and weights must have the same length"

coms = set()
graphs = []
for path in args.graphs:
    graph = json.load(open(path))
    coms.update(graph)
    graphs.append(graph) 

def avg(scores):
    score, weight = zip(*scores)
    score = np.array(score)
    weight = np.array(weight)
    return (score * weight).sum()/weight.sum()

merge = {}
for com in tqdm(coms, desc="Merging"):
    if com == '':
        continue
    term_to_scores = defaultdict(list)
    for graph, weight in zip(graphs, args.weights):
        try:
            for term, score in graph[com].items():
                term_to_scores[term.replace('_',' ')].append((score, weight))
        except:
            pass
    
    merge[com] = {term:avg(scores) for term, scores in term_to_scores.items() }

json.dump(merge, open(args.output,'w'))