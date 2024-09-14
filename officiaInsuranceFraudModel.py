import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
data = pd.read_excel("healthcare_claims_data.xlsx")

# Convert categorical variables to category type
data['Procedure Type'] = data['Procedure Type'].astype('category')
data['Patient Risk Category'] = data['Patient Risk Category'].astype('category')
data['Claim ID'] = data['Claim ID'].astype('category')

# Define fraud based on conditions
conditions = [
    (data['Patient Risk Category'] == 'Low') & (data['Procedure Type'] == 'Invasive'),
    (data['Procedure Type'] == 'Diagnostic') & (data['Cost'] > 1000),
    (data['Number of Procedures'] > 3)
]
data['Fraud'] = np.select(conditions, [1, 1, 1], default=0)

# Count the number of duplicate Claim IDs
duplicate_claim_ids = data['Claim ID'].duplicated().sum()
print("Number of duplicate Claim IDs:", duplicate_claim_ids)

# Display the updated DataFrame
print(data['Fraud'])

# Define a custom color palette
light_blue = (0.5, 0.7, 0.9)  # RGB values for light blue
light_red = (1.0, 0.4, 0.4)   # RGB values for light red
custom_palette = {0: light_blue, 1: light_red}

# Display pairplot
sns.set(style="whitegrid")
sns.pairplot(
    data[["Claim ID", "Cost", "Procedure Type", "Number of Procedures", "Fraud"]],
    hue="Fraud",
    height=3,
    palette=custom_palette
)
plt.show()

# Exclude non-numeric columns before creating the first heatmap
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
sns.heatmap(data[numeric_columns].corr(), annot=True)
plt.show()

# Perform one-hot encoding for categorical variables excluding the target variable "Fraud"
data_encoded = pd.get_dummies(data.drop(columns=['Fraud']))

# Add the target variable back to the encoded DataFrame
data_encoded['Fraud'] = data['Fraud']

# Separate features and target variable
X = data_encoded.drop("Fraud", axis=1)
y = data_encoded["Fraud"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create and train the logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Plotting the predicted values for all cases
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_pred)), y_pred, c='b', label='Predicted values', alpha=0.5)
plt.scatter(range(len(y_test)), y_test, c=y_test, cmap='coolwarm', label='Actual values', alpha=0.5)
plt.xlabel('Sample index')
plt.ylabel('Fraud (0: No Waste, 1: Waste Detected)')
plt.title('Predicted vs Actual Values for All Cases')
plt.legend()
plt.show()

# Identify predicted positive cases
predicted_positive_indices = y_pred == 1

# Compare with actual negative cases
actual_negative_indices = y_test == 0

# Count false positives
false_positives = sum(predicted_positive_indices & actual_negative_indices)

print("Number of false positives:", false_positives)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained logistic regression model
joblib.dump(lr, 'officialHC_fraud_model.pkl')
