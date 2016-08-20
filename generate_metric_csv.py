import nltk
#from nltk.corpus import gutenberg
from stylometry.extract import *
#import os
#import psycopg2



def main():

	
	## Read all files in the directory, calculate metrics and write all to a csv file
	all_files = StyloCorpus.from_glob_pattern('/home/vagrant/books/split/*')
	all_files.output_csv('/home/vagrant/all_files.csv')


if __name__ == "__main__":
    main()

