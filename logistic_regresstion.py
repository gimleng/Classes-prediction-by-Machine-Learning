import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('project_datas\meter_replace.csv',encoding = 'utf-8')

#Data Frame without row that contain 999 value in 'check_result'
# df_no999 = df[df.check_result != 999]
# df.head()
# df.info()

# Convert features of each dataframe to list
X = df[['meter_age', 'prev_change5', 'prev_change6', 'prev_change7', 'prev_change8', 'prev_change9']].values.tolist()

# Convert value in "check_result" table in to list
y = df['check_result'].values.tolist()

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=20)
# show result of above function
# print(f"X_train: {X_train},\nX_test: {X_test},\ny_train: {y_train},\ny_test: {y_test}")

# Store data for test 0.30 (30%)
# X total --> len(X_train + X_test) = 4,025
# X_train --> len(X_train) = 2,817 (70%)
# X_test --> len(X_test) = 1,208 (30%)


X, y = make_multilabel_classification(n_classes=10)
clf = LogisticRegression( solver='saga' , max_iter=10000, random_state=20).fit(X_train, y_train)

#clf_coef = clf.coef_[0]

# y predicted variable
y_pred = clf.predict(X_test)

# Show the classification report
print(classification_report(y_test, y_pred))

# Create confusion matrix to compare y_test and y_pred
cm = confusion_matrix(y_test, y_pred)

# Create DataFrame from confusion matrix
cm_df = pd.DataFrame(cm,
                     index = ['1', '2', '3', '4', '5', '6', '7', '8', '9','10', '999'],
                     columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '999'])

#Plotting the result from Confusion matrix 
plt.figure(figsize=(12, 8))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actual values')
plt.xlabel('Predicted values')
plt.show() 
 
