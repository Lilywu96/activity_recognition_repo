#! /usr/bin/env python
'''
Pass in output of file3.py  E.g.
cat file3_name | python -m huyun_preprocessing.file4

Output is original time series label along with cluster labels for each window. E.g.
ts_label, c1, c3, c1, ....
'''
import sys

import config

if __name__ == '__main__':
  time_series = []
  current_label = None
  for line in sys.stdin:
    (label, cluster) = line.strip().split(config.OUTPUT_DELIMITER)
    if current_label != label:
      #I feel bad for doing this
      if time_series != []:
        print ','.join(time_series)
      current_label = label
      time_series = [label]
    time_series.append(cluster)
  print ','.join(time_series)
