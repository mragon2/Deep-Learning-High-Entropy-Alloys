#!/usr/bin/python

'''
Author: Marco Ragone, Computational Multiphase Transport Laboratory, University of Illinois at Chicago
'''

from scripts.training_utils import*


if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument('-j', type=str, dest='json_path', help='Path to configuration .json file')
	args = parser.parse_args()

	config = Config(args.json_path)

	#train_serial(config)
	train_model_parallel(config)
	#train_data_parallel(config)
