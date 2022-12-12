#parse arguments
import argparse


def parse_args():
	my_parser = argparse.ArgumentParser()

	my_parser.add_argument('-original', help='path to data file')

	my_parser.add_argument('-transfer', help='transfer model')

	my_parser.add_argument('-test', help='transfer model')

	args = my_parser.parse_args()

	print('Entered args')

	return args


