#parse arguments
import argparse


def parse_args():
	my_parser = argparse.ArgumentParser()

	my_parser.add_argument('-d', help='path to data file')

	args = my_parser.parse_args()

	print('Entered args')

	return args


