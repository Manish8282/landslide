mport numpy as np
import pandas as pd
import wandb

data = pd.read_csv("bhutan_landslide_data.csv")
data.drop(['FID', 'Type', 'TWI'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split
# Split the data into X and y
X = data.drop('Code', axis=1)
y = data['Code']
#split the data in training and remaining dataset
X_train, X_rem, y_train, y_rem = train_test_split(X,y, stratify=y, train_size=0.7, random_state=20)
#split remaining in Validation and test
test_size = 0.3
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, stratify=y_rem, test_size=0.3, random_state=20)

#Train with Random Forrest
from sklearn.ensemble import RandomForestClassifier

wandb.init(project = 'Bhutan_landslide_prediction', entity='manishrai727', name='RandomForestClassifier')

rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train
rf.fit(X_train, y_train)

# prediction
y_pred = rf.predict(X_valid)

# Evaluate the model's accuracy on the testing set
accuracy = accuracy_score(y_valid, y_pred)
precision = precision_score(y_valid, y_pred)
recall = recall_score(y_valid, y_pred)
f1 = f1_score(y_valid, y_pred)

wandb.log({'accuracy': accuracy, 'precision': precision, 'recall': recall})
