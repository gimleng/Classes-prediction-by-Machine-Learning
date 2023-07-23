import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('project_datas/meter_replace.csv', encoding='utf-8')
#df = df[df.check_result != 999]
X = df[['meter_age', 'prswtusg', 'prev_change1', 'prev_change2', 'prev_change3', 'prev_change4',      'prev_change5', 'prev_change6', 'prev_change7', 'prev_change8', 'prev_change9']]#.values.tolist()
y = df['check_result']#.values.tolist()

data = pd.concat([X, df['check_result']], axis=1)

#data, which drop index:2021 to remove outlier
#data = pd.concat([X.drop([2021]), df['check_result']], axis=1)
"""
def plot_box(data, cols, col_x = 'check_result'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.boxplot(x=col_x, y=col, data = data)
        plt.xlabel(col_x)
        plt.ylabel(col)
        plt.show()
"""
num_cols = ['meter_age', 'prswtusg', 'prev_change1', 'prev_change2', 'prev_change3', 'prev_change4', 
        'prev_change5', 'prev_change6', 'prev_change7', 'prev_change8', 'prev_change9']
#plot_box(data, num_cols)

# class2_outlier = X[(X['prev_change5'] > 2000)]
# result: index 2021


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)

rfc=RandomForestClassifier(random_state=30)

param_grid = { 
 'n_estimators': [100,200,300,400,500,600], #100,200,300,400,500,600
 'max_features': ['sqrt', 'log2'], #'auto', 'sqrt', 'log2'
 'max_depth' : [2,3,4,5,6,7,8,9,10], #2,3,4,5,6,7,8,9,10
 'criterion' :['gini', 'entropy'] #'gini', 'entropy'
}

#GridSearchCV
#CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
#CV_rfc.fit(X_train, y_train)
#print(CV_rfc.best_estimator_)

#Random Search
#random_search = RandomizedSearchCV(estimator=rfc, param_distributions=param_grid, cv=5)
#random_search.fit(X_train, y_train)
#print(random_search.best_estimator_)

#Select parameters randomly
rfc1 = RandomForestClassifier(random_state = 30, max_features= None, n_estimators = 500, max_depth = 7, criterion = 'gini')
rfc1.fit(X_train, y_train)
pred1=rfc1.predict(X_test)
print("Accuracy rfc1 : ",accuracy_score(y_test,pred1))

#From Random search
rfc2 = RandomForestClassifier(criterion='entropy', max_depth=5, max_features='log2',n_estimators=300, random_state=30)
rfc2.fit(X_train, y_train)
pred2=rfc2.predict(X_test)
print("Accuracy rfc2 : ",accuracy_score(y_test,pred2))

#From Random search
rfc3 = RandomForestClassifier(criterion='entropy', max_depth=4, max_features='log2',n_estimators=600, random_state=30)
rfc3.fit(X_train, y_train)
pred3=rfc3.predict(X_test)
print("Accuracy rfc3 : ",accuracy_score(y_test,pred3))

print(classification_report(y_test,pred1))

"""
cm = confusion_matrix(y_test, pred1)
cm_df = pd.DataFrame(cm,
                     index = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
                     columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
#Plotting the result from Confusion matrix 
plt.figure(figsize=(12, 8))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actual values')
plt.xlabel('Predicted values')
plt.show()
"""