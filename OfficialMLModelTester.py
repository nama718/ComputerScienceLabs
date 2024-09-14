import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import numpy as np

# Load the new dataset
new_data = pd.read_excel("predictionHC_claims_data.xlsx")

# Exclude the Claim ID feature
X_new = new_data.drop("Claim ID", axis=1)

# Define fraud based on conditions
conditions = [
    (new_data['Patient Risk Category'] == 'Low') & (new_data['Procedure Type'] == 'Invasive'),
    (new_data['Procedure Type'] == 'Diagnostic') & (new_data['Cost'] > 1000),
    (new_data['Number of Procedures'] > 3)
]
new_data['Fraud'] = np.select(conditions, [1, 1, 1], default=0)



# Load the trained model
loaded_model = joblib.load('officialHC_fraud_model.pkl')

# Use the trained model to make predictions
y_pred_new = loaded_model.predict(X_new)

# Evaluate the predictions
accuracy = accuracy_score(new_data['Fraud'], y_pred_new)
conf_matrix = confusion_matrix(new_data['Fraud'], y_pred_new)
classification_rep = classification_report(new_data['Fraud'], y_pred_new)

print("Accuracy on the new dataset:", accuracy)
print("\nConfusion Matrix on the new dataset:\n", conf_matrix)
print("\nClassification Report on the new dataset:\n", classification_rep)
