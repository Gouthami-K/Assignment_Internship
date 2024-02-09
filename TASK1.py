import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

train_data=pd.read_excel('train.xlsx')
test_data = pd.read_excel('test.xlsx') 

print(train_data.head())
print(test_data.head())

#categorical_feature=[features for features in train_data.columns if train_data[features].dtype == "O"]
#categorical_feature

#en = LabelEncoder()
#for cols in categorical_feature:
#    train_data[cols] = en.fit_transform(train_data[cols])

# Split the data into independent and dependent features
X = train_data.drop("target",axis=1)
y = train_data["target"]

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

# Create a Scaler instance
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_train = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_val = scaler.transform(X_val)

# Create and train a RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

# Accuracy Train set
train_predictions_rf = model.predict(X_train)

# target values for the Validation set
print('Train Predictions:', train_predictions_rf)

# Evaluate the model accuracy on the training set
train_accuracy = accuracy_score(y_train, train_predictions_rf)
print(f'Train Accuracy: {train_accuracy}')


# Accuracy Validation set
val_predictions_rf = model.predict(X_val)

# target values for the Validation set
print('Val Predictions:', val_predictions_rf)

accuracy = accuracy_score(y_val, val_predictions_rf)
print(f'Model Accuracy on Validation Set: {accuracy}')

# TEST DATA

# Scale the test data using the same scaler
X_test = scaler.transform(test_data)

test_predictions_rf = model.predict(X_test)

# target values for the test set
print('Test Predictions:', test_predictions_rf)

predictions_df = pd.DataFrame({'Predicted_Target': test_predictions_rf})
predictions_df.to_csv('predictions.csv', index=False)


