# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

# load data
url = "/Users/pmcclarence/Vagrantboxes/GW/vagrant/training_dataset.csv"

names = ['a.LexicalDiversity','a.MeanWordLen','a.MeanSentenceLen','a.StdevSentenceLen','a.MeanParagraphLen','a.DocumentLen','a.Commas','a.Semicolons','a.Quotes','a.Exclamations','a.Colons','a.Dashes','a.Mdashes','a.Ands','a.Buts','a.Howevers','a.Ifs','a.Thats','a.Mores','a.Musts','a.Mights','a.This','a.Verys',
'b.LexicalDiversity','b.MeanWordLen','b.MeanSentenceLen','b.StdevSentenceLen','b.MeanParagraphLen','b.DocumentLen','b.Commas','b.Semicolons','b.Quotes','b.Exclamations','b.Colons','b.Dashes','b.Mdashes','b.Ands','b.Buts','b.Howevers','b.Ifs','b.Thats','b.Mores','b.Musts','b.Mights','b.This','b.Verys', 'Output']

dataframe = pandas.read_csv(url, names=names, delimiter='|')
array = dataframe.values
X = array[:,0:46]
Y = array[:,46]
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:46,:])


# feature extraction
print "\nRecursive feature elimination"
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_

# feature extraction
print "\nPrincipal componant analysis"
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s") % fit.explained_variance_ratio_
print(fit.components_)


# feature extraction
print "\nBagged decision trees"
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)
