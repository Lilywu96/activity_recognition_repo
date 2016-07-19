#! /usr/bin/env python
'''
Pass in output of file2.py E.g.
python -m preprocessing.file3 <file2_name>
Output:
time_series_label, kmeans_window_label

Create label data using old cluster
'''
import sys
import pandas as pd
import scipy.cluster.vq as vq
import pickle

import config

if __name__ == '__main__':
  #No way to be memory efficient here
  df = pd.read_csv(sys.argv[1], sep = config.OUTPUT_DELIMITER, header = None)
  labels = df[df.columns[0]]
  samples = df[df.columns[1:]]
  #No need to whiten for kmeans2
  f = open(config.CENTER_FILE, 'rb')
  km_centers = pickle.load(km_centers)
  f.close()
  (km_labels, junk) = vq.vq(samples, km_centers)
  for i in range(0,len(labels)):
    print '{0},{1}'.format(labels[i], km_labels[i])
