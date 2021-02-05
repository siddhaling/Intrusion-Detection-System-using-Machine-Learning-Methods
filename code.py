import pandas
import numpy
import warnings
import itertools
import matplotlib.pyplot as plt
import seaborn

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics as metrics

warnings.filterwarnings("ignore")

train = pandas.read_csv("train.csv")
test = pandas.read_csv("test.csv")

print(train.head())

print("Training data has {} rows & {} columns".format(train.shape[0],train.shape[1]))

print(test.head())

print("Testing data has {} rows & {} columns".format(test.shape[0],test.shape[1]))

train.describe()

ratio = train['class'].value_counts()
labels = ratio.index[0], ratio.index[1]
sizes = [ratio.values[0], ratio.values[1]]

figure, axis = plt.subplots()
axis.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
axis.axis('equal')

plt.show()

print(train['num_outbound_cmds'].value_counts())
print(test['num_outbound_cmds'].value_counts())

train.drop(['num_outbound_cmds'], axis=1, inplace=True)
test.drop(['num_outbound_cmds'], axis=1, inplace=True)

scaler = StandardScaler()

cols = train.select_dtypes(include=['float64','int64']).columns
sc_train = scaler.fit_transform(train.select_dtypes(include=['float64','int64']))
sc_test = scaler.fit_transform(test.select_dtypes(include=['float64','int64']))

sc_traindf = pandas.DataFrame(sc_train, columns = cols)
sc_testdf = pandas.DataFrame(sc_test, columns = cols)

encoder = LabelEncoder()

cattrain = train.select_dtypes(include=['object']).copy()
cattest = test.select_dtypes(include=['object']).copy()

traincat = cattrain.apply(encoder.fit_transform)
testcat = cattest.apply(encoder.fit_transform)

enctrain = traincat.drop(['class'], axis=1)
cat_Ytrain = traincat[['class']].copy()

train_x = pandas.concat([sc_traindf,enctrain],axis=1)
train_y = train['class']
train_x.shape

test_df = pandas.concat([sc_testdf,testcat],axis=1)
test_df.shape

rfc = RandomForestClassifier();

rfc.fit(train_x, train_y);

score = numpy.round(rfc.feature_importances_,3)
importances = pandas.DataFrame({'feature':train_x.columns,'importance':score})
importances = importances.sort_values('importance',ascending=False).set_index('feature')

plt.rcParams['figure.figsize'] = (16,4)
importances.plot.bar();

rfc = RandomForestClassifier()

rfe = RFE(rfc, n_features_to_select=10)
rfe = rfe.fit(train_x, train_y)

feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), train_x.columns)]
selected_features = [v for i, v in feature_map if i==True]

print(selected_features)

seaborn.heatmap(train_x[selected_features].corr(), annot = True, fmt='.1g')

X_train,X_test,Y_train,Y_test = train_test_split(train_x,train_y,train_size=0.60, random_state=2)

model = KNeighborsClassifier(n_jobs=-1)
model.fit(X_train, Y_train);

scores = cross_val_score(model, X_train, Y_train, cv=10)
accuracy = metrics.accuracy_score(Y_train, model.predict(X_train))
confusion_matrix = metrics.confusion_matrix(Y_train, model.predict(X_train))
classification = metrics.classification_report(Y_train, model.predict(X_train))

print ("Cross Validation Mean Score:" "\n", scores.mean())
print ("Model Accuracy:" "\n", accuracy)
print ("Confusion matrix:" "\n", confusion_matrix)
print ("Classification report:" "\n", classification)

accuracy = metrics.accuracy_score(Y_test, model.predict(X_test))
confusion_matrix = metrics.confusion_matrix(Y_test, model.predict(X_test))
classification = metrics.classification_report(Y_test, model.predict(X_test))
                                                                     
print ("Model Accuracy:" "\n", accuracy)
print ("Confusion matrix:" "\n", confusion_matrix)
print ("Classification report:" "\n", classification)

prediction = model.predict(test_df)
test['prediction'] = prediction
print(test.head())

prediction

ratio = test['prediction'].value_counts()
labels = ratio.index[0], ratio.index[1]
sizes = [ratio.values[0], ratio.values[1]]

figure, axis = plt.subplots()
axis.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
axis.axis('equal')

plt.show()