'''
Config file
'''
WINDOW_SIZE = 48
NUM_CLUSTERS = 200
LABEL_MAP = {'32' : '-1', '48' : '0', '49' : '1', '50' : '2', '51' : '3', '52' : '4', '53' : '5', '54' : '6', '55' : '7', '56' : '8', '57' : '9'}
DELIMITER = ' '
OUTPUT_DELIMITER = ','
FREQUENCY_VALUE_BINS = 10 #for value discretization
CENTER_FILE = 'centers.pkl.temp'

HUYUN_OUTPUT_DELIMITER = ' '
HUYUN_NUM_CLUSTERS = 200
HUYUN_CENTER_FILE = 'huyun_centers.pkl.temp'
#Join 3 original samples into 1.
HUYUN_JOIN_VALUE = 3

PART2_WINDOW = 40
PART2_DELIMITER = ','
PART2_SHIFT = 0.5

