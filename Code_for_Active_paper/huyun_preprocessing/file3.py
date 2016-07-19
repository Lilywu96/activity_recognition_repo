#! /usr/bin/env python
'''
Pass in huyun sensor data E.g.
python -m huyun_preprocessing.file3 data/huynh/train.txt
Output:
time_series_label, kmeans_window_label

Create new clusters using input data
'''
import sys
import pandas as pd
import scipy.cluster.vq as vq
import pickle

import config

if __name__ == '__main__':
  #No way to be memory efficient here
  df = pd.read_csv(sys.argv[1], sep = config.HUYUN_OUTPUT_DELIMITER, header = None)
  labels = df[df.columns[0]]
  samples = df[df.columns[1:-2]]
  #No need to whiten for kmeans2
  (km_centers, km_labels) = vq.kmeans2(samples.values,config.HUYUN_NUM_CLUSTERS)
  f = open(config.HUYUN_CENTER_FILE, 'wb')
  pickle.dump(km_centers, f)
  f.close()
  for i in range(0,len(labels)):
    print '{0},{1}'.format(labels[i], km_labels[i])
