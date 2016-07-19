#! /usr/bin/env python
'''
Pass in activity primitive representation. E.g.
cat file4 | python -m part2.window2
'''
import sys
import config

def arff_header(window_size):
  print '@Relation test'
  print '@ATTRIBUTE label {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}'
  for i in range(0,window_size):
    print '@ATTRIBUTE a{0} NUMERIC'.format(i)
  print '@DATA'

if __name__ == '__main__':
  window_size = config.PART2_WINDOW
  shift_size = int(window_size*config.PART2_SHIFT)
  arff_header(window_size)
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
        line.extend(block)
        print config.PART2_DELIMITER.join(line)
        start = start + shift_size
        end = end + shift_size
    #Last block. Include [end - window_size:end]
#    block = vect[n - window_size:n]
#    line = [label]
#    line.extend(block)
#    print config.PART2_DELIMITER.join(line)
