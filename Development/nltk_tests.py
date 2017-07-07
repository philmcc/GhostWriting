import nltk
from nltk import word_tokenize
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from gutenberg.query import get_etexts
from gutenberg.query import get_metadata
import sys

for x in range (1000, 20000):
	try:
		text = strip_headers(load_etext(x)).strip()
		#print(text)  # prints 'MOBY DICK; OR THE WHALE\n\nBy Herman Melville ...'

		author = get_metadata('author', x)
		print author
		author_name = list(author)
		author_name = str(author_name[0])
		author_name = author_name.split(',')
		author_name_clean = str(author_name[0]).strip() + str(author_name[1]).strip()
		author_name_clean = author_name_clean.replace(" ", "")
		print author_name_clean



		filename = "books/"+ author_name_clean + ".txt"
		fo = open(filename, "w")

		fo.write(str(text))
		fo.close()
	except:
	    print("Unexpected error:", sys.exc_info()[0])