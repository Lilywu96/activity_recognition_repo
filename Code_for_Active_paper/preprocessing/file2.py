#! /usr/bin/env python
'''
Pass in sensor file. E.g.
cat sensor16.txt | python -m preprocessing.file2
'''
import sys
import pandas as pd
import numpy as np
import scipy.stats as stats

import config
import pickle

def extract_statistical_features(series):
    val = []
    val.append(series.mean())
    val.append(series.std())
    val.append(series.var())
    val.append(series.min())
    val.append(series.max())
    val.append(np.mean(series**2))
    val.append(stats.entropy(np.histogram(series, bins=config.FREQUENCY_VALUE_BINS, density=True)[0]))
    return val

def process_time_series(time_series, label, window_size):
  if len(time_series) == 0:
    return None
  shift_size = int(window_size*0.5)
  label = config.LABEL_MAP[label]
  n = len(time_series)
  start = 0
  end = start + window_size
  if window_size != -1:
    while end < n:
      block = time_series[start:end]
      statistical_features = [str(x) for x in process_time_series_block(block)]
      line = [label]
      line.extend(statistical_features)
      print config.OUTPUT_DELIMITER.join(line)
      start = start + shift_size
      end = end + shift_size
  #Last block has length < window_size. Include it for now.
  block = time_series[start:n]
  statistical_features = [str(x) for x in process_time_series_block(block)]
  line = [label]
  line.extend(statistical_features)
  print config.OUTPUT_DELIMITER.join(line)
  
def process_time_series_block(block):
  df = pd.DataFrame(block)
  val = []
  for i in range(0,df.shape[1]):
    val.extend(extract_statistical_features(df[i]))
  return val

if __name__ == '__main__':
  time_series = []
  current_label = None
  for line in sys.stdin:
    parts = line.strip().split(config.DELIMITER)
    label = parts[0]
    vect = map(lambda(x): int(x),parts[1:])
    if current_label != label:
      process_time_series(time_series, current_label, config.WINDOW_SIZE)
      current_label = label
      time_series = []
    time_series.append(vect)
  process_time_series(time_series, current_label, config.WINDOW_SIZE)
  lengths.append(len(time_series))
