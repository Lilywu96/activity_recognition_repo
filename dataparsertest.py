
import unittest
from dataparser import DataParser, Session

__author__ = 'alexander'

class TestSampleParsing(unittest.TestCase):
    def testBasicFileParsing(self):
        dataParser = DataParser()
        dataParser.parseFile("/Users/LilyWU/Documents/PAMAP/PAMAP2_Dataset/Protocol/subject101.dat")
        # for sess in dataParser.sessions:
        #   #Session()=session
        #   for sample in sess[1]:
        #      print(1,sample.samples.hand.accX)
        print(dataParser.sessions[1].samples.hand.accX)
if __name__ == '__main__':
    test=TestSampleParsing()
    test.testBasicFileParsing()