import nltk

from stylometry.extract import *




def main():

	filename = '/Users/pmcclarence/philmccgit/GhostWriting/txtfiles/Processed/BramStoker.txt'
	

	
	textfile = StyloDocument(filename)
	textfile.text_output()

if __name__ == "__main__":
    main()

