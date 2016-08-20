import nltk
from stylometry.extract import *
import sys
import pandas
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import pickle


def main(argv):

	
	file1 = argv[0]
	file2 = argv[1]
	#print file1 + " " + file2

	file1 = '/Users/pmcclarence/philmccgit/GhostWriting/txtfiles/split/' + file1
	file2 = '/Users/pmcclarence/philmccgit/GhostWriting/txtfiles/split/' + file2
	#pwdfile2 = '/Users/pmcclarence/philmccgit/GhostWriting/JMBarrie_aa'

	#files = np.array([{'JamesJoyce_ec','JaneAusten_cg','1'},{'HermanMelville_ak','HermanMelville_am','0'}])


	#file1 = '/Users/pmcclarence/philmccgit/GhostWriting/txtfiles/split/' + a[0]
	#file2 = '/Users/pmcclarence/philmccgit/GhostWriting/txtfiles/split/' + b[0]
	#outbut = c
	file1_metrics = StyloDocument(file1)
	file2_metrics = StyloDocument(file2)
	#test_array= np.array([[file1_metrics.type_token_ratio(),file1_metrics.mean_word_len(),file1_metrics.mean_sentence_len(),file1_metrics.std_sentence_len(),file1_metrics.mean_paragraph_len(),file1_metrics.document_len(),file1_metrics.term_per_thousand(','),file1_metrics.term_per_thousand(';'),file1_metrics.term_per_thousand('"'),file1_metrics.term_per_thousand('!'),file1_metrics.term_per_thousand(':'),file1_metrics.term_per_thousand('-'),file1_metrics.term_per_thousand('--'),file1_metrics.term_per_thousand('and'),file1_metrics.term_per_thousand('but'),file1_metrics.term_per_thousand('however'),file1_metrics.term_per_thousand('if'),file1_metrics.term_per_thousand('that'),file1_metrics.term_per_thousand('more'),file1_metrics.term_per_thousand('must'),file1_metrics.term_per_thousand('might'),file1_metrics.term_per_thousand('this'),file1_metrics.term_per_thousand('very'),file2_metrics.type_token_ratio(),file2_metrics.mean_word_len(),file2_metrics.mean_sentence_len(),file2_metrics.std_sentence_len(),file2_metrics.mean_paragraph_len(),file2_metrics.document_len(),file2_metrics.term_per_thousand(','),file2_metrics.term_per_thousand(';'),file2_metrics.term_per_thousand('"'),file2_metrics.term_per_thousand('!'),file2_metrics.term_per_thousand(':'),file2_metrics.term_per_thousand('-'),file2_metrics.term_per_thousand('--'),file2_metrics.term_per_thousand('and'),file2_metrics.term_per_thousand('but'),file2_metrics.term_per_thousand('however'),file2_metrics.term_per_thousand('if'),file2_metrics.term_per_thousand('that'),file2_metrics.term_per_thousand('more'),file2_metrics.term_per_thousand('must'),file2_metrics.term_per_thousand('might'),file2_metrics.term_per_thousand('this'),file2_metrics.term_per_thousand('very')]])
	test_array= np.array([['1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1']])
	
	print test_array
	my_array = pandas.DataFrame(test_array)
	print "\nData Shape: "
	print(my_array.shape)

	#test_array= np.array([[file2_metrics.type_token_ratio(),file2_metrics.mean_word_len(),file2_metrics.mean_sentence_len(),file2_metrics.std_sentence_len(),file2_metrics.mean_paragraph_len(),file2_metrics.document_len(),file2_metrics.term_per_thousand(','),file2_metrics.term_per_thousand(';'),file2_metrics.term_per_thousand('"'),file2_metrics.term_per_thousand('!'),file2_metrics.term_per_thousand(':'),file2_metrics.term_per_thousand('-'),file2_metrics.term_per_thousand('--'),file2_metrics.term_per_thousand('and'),file2_metrics.term_per_thousand('but'),file2_metrics.term_per_thousand('however'),file2_metrics.term_per_thousand('if'),file2_metrics.term_per_thousand('that'),file2_metrics.term_per_thousand('more'),file2_metrics.term_per_thousand('must'),file2_metrics.term_per_thousand('might'),file2_metrics.term_per_thousand('this'),file2_metrics.term_per_thousand('very')]])
	

	

	model_name =  'stepbystep/finalized_model.sav'
	loaded_model = pickle.load(open(model_name,  'rb' ))
	scaler_same = StandardScaler().fit(my_array)
	rescaledX_same = scaler_same.transform(my_array)
	result = loaded_model.predict(my_array)
	#result = loaded_model.score(my_array,0)
	print(loaded_model.predict(my_array))
	if result == 1:
		print "\n Different Authors\n"
		print("The Result is: " + str(result))
	else:
		print "\n Same Authors\n"
		print("The Result is: " + str(result))

if __name__ == "__main__":
    main(sys.argv[1:])

#JamesJoyce_db
#JamesJoyce_dv

#JaneAusten_bm
#JaneAusten_cf

#JMBarrie_ae
#JMBarrie_aa