#!/usr/bin/python

'''
Author: Marco Ragone, Computational Multiphase Transport Laboratory, University of Illinois at Chicago
'''

import os

if __name__ == '__main__':

	#os.system('python scripts/training.py')
	#os.system('python scripts/training_data_parallelization.py')
	os.system('horovodrun -np 1 -H localhost:1 python scripts/training_model_parallelization.py')


