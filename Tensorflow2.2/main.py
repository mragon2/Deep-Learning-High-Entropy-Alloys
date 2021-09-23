#!/usr/bin/python

import os
if __name__ == '__main__':

	os.system('python training.py')
	#os.system('python training_data_parallelization.py')
	#os.system('horovodrun -np 1 -H localhost:1 python training_model_parallelization.py')


