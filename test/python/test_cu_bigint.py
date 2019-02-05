
import os,sys, os.path
import numpy as np

from random import randint

sys.path.append(os.path.abspath(os.path.dirname('../../lib/')))
import cu_bigint


sys.path.append('../../src/python')
from bigint import *

def test():
    max_num = (1<<256) - 1
    bn = [BigInt(randint(0,max_num)) for x in range(100)]
    bn256 = [n.as_uint256() for n in bn]
    

    bn_vector = cu_bigint.BigInt(bn256)
    bn_vector.addm()
    
    results2 = bn_vector.retreive()


if __name__ == "__main__":
  test()
