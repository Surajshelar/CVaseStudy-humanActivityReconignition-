import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm

df = pd.read_csv('train.csv')

lnr = LogisticRegression(random_state=0)
rfc = RandomForestClassifier(random_state=1)
dtc = DecisionTreeClassifier()
gbc = GradientBoostingClassifier(n_estimators=10)
sc  = svm.SVC()

df = pd.read_csv('train.csv')

x = df.drop(['Activity', 'subject'], axis=1)
y = df['Activity'].astype(object)

le = LabelEncoder()
y = le.fit_transform(y)

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=0, train_size=0.3)

lnr.fit(x_train, y_train)
rfc.fit(x_train, y_train)
dtc.fit(x_train, y_train)
gbc.fit(x_train, y_train)
sc.fit(x_train, y_train)

lnr_predict = lnr.predict(x_test)
rfc_predict = rfc.predict(x_test)
dtc_predict = dtc.predict(x_test)
gbc_predict = gbc.predict(x_test)
sc_predict = sc.predict(x_test)

print('LogisticRegression', accuracy_score(y_test, lnr_predict))
print('RandomForest', accuracy_score(y_test, rfc_predict))
print('DecisionTree', accuracy_score(y_test, dtc_predict))
print('GradientBoostingClassifier', accuracy_score(y_test, gbc_predict))
print('SVM', accuracy_score(y_test, sc_predict))

#LogisticRegression 0.9714396735962697
#RandomForest 0.9677482028366039
#DecisionTree 0.914319020788809
#GradientBoostingClassifier 0.9411307557800661
#SVM 0.9638624441422188