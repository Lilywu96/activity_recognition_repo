#! /usr/bin/env python
'''
Pass in activity primitive representation. E.g.
cat file4 | python -m part2.tf
'''
import sys
import config

def arff_header(num_clusters):
  print '@Relation test'
#Skoda dataset
#  print '@ATTRIBUTE label {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}'
#Huyun dataset
  print '@ATTRIBUTE label {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37}'
  for i in range(0,num_clusters):
    print '@ATTRIBUTE a{0} NUMERIC'.format(i)
  print '@DATA'

'''
Return frequencies.
1,1,2 -> [2/3, 1/3, 0, 0, ..., 0] 
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
  return_vect = ['0']*num_clusters
  for key in freq_dict:
    return_vect[int(key)] = str(freq_dict[key]/total)
  return return_vect
    
if __name__ == '__main__':
  window_size = config.PART2_WINDOW
  shift_size = int(window_size*config.PART2_SHIFT)
  arff_header(config.NUM_CLUSTERS)
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
      block = vect[start:end]
      if len(block) == window_size:
        line = [label]
        block = process_vect(block)
        line.extend(block)
        print config.PART2_DELIMITER.join(line)
        start = start + shift_size
        end = end + shift_size
    #Last block. Include [end - window_size:end]
#    block = vect[n - window_size:n]
#    line = [label]
#    line.extend(block)
#    print config.PART2_DELIMITER.join(line)
