#! /usr/bin/env python
'''
Pass in activity primitive representation. E.g.
python -m part2.tfidf file4
'''
import sys
import config
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pandas as pd

def arff_header(num_clusters):
  print '@Relation test'
  print '@ATTRIBUTE label {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}'
  for i in range(0,num_clusters):
    print '@ATTRIBUTE a{0} NUMERIC'.format(i)
  print '@DATA'

'''
Return raw counts.
1,1,2 -> [2, 1, 0, 0, ..., 0] 
Pads with config.NUM_CLUSTERS 0's.
'''
def process_vect(vect):
  freq_dict = {}
  total = config.PART2_WINDOW
  num_clusters = config.NUM_CLUSTERS
  for val in vect:
    if val not in freq_dict:
      freq_dict[val] = 0.0
    freq_dict[val] = freq_dict[val] + 1
  return_vect = [0.0]*num_clusters
  for key in freq_dict:
    return_vect[int(key)] = freq_dict[key]
  return return_vect
    
if __name__ == '__main__':
  transformer = TfidfTransformer()
  
  window_size = config.PART2_WINDOW
  shift_size = int(window_size*config.PART2_SHIFT)
  arff_header(config.NUM_CLUSTERS)
  matrix = []
  labels = []
  for line in sys.stdin:
    parts = line.strip().split(config.PART2_DELIMITER)
    label = parts[0]
    vect = parts[1:]
    #Break activity primitive vector into sliding windows
    window_size = config.PART2_WINDOW
    n = len(vect)
    start = 0
    end = start + window_size
    while end <= n:
      labels.append(label)
      block = vect[start:end]
      if len(block) == window_size:
#        line = [int(label)]
        block = process_vect(block)
#        line.extend(block)
        matrix.append(block)
        start = start + shift_size
        end = end + shift_size
  matrix = np.array(matrix)
  tfidf = transformer.fit_transform(matrix)
  tfidf = tfidf.toarray()
  for i in range(0,matrix.shape[0]):
    print labels[i] + ',' + ','.join([str(x) for x in tfidf[i].tolist()])
    #Last block. Include [end - window_size:end]
#    block = vect[n - window_size:n]
#    line = [label]
#    line.extend(block)
#    print config.PART2_DELIMITER.join(line)
