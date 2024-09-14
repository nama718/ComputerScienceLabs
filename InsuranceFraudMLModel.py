import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
data = pd.read_excel("InsuranceFraud.xlsx")

# Convert categorical variables to category type
data['Patient ID'] = data['Patient ID'].astype('category')
data['Provider ID'] = data['Provider ID'].astype('category')
data['Diagnosis Code'] = data['Diagnosis Code'].astype('category')
data['Procedure Code'] = data['Procedure Code'].astype('category')
data['Claim ID'] = data['Claim ID'].astype('category')

# Creating a new column based on existing columns
data['Balance Difference'] = data['Billing Amount'] - data['Paid Amount']
data['Service Date'] = pd.to_datetime(data['Service Date'])
data['Claim Submission Date'] = pd.to_datetime(data['Claim Submission Date'])

# Calculate the time difference
data['Time Difference'] = (data['Claim Submission Date'] - data['Service Date']).dt.days / 30

# Define fraud based on conditions
data['Fraud'] = ((data['Time Difference'] > 3) |
                 (data['Billing Amount'] < 3 * data['Balance Difference']))

# Count the number of duplicate patient IDs
duplicate_patient_ids = data['Patient ID'].duplicated().sum()
print("Number of duplicate patient IDs:", duplicate_patient_ids)

# Count the number of duplicate procedure codes
duplicate_procedure_codes = data['Procedure Code'].duplicated().sum()
print("Number of duplicate procedure codes:", duplicate_procedure_codes)

# Count the number of duplicate diagnosis codes
duplicate_diagnosis_codes = data['Diagnosis Code'].duplicated().sum()
print("Number of duplicate diagnosis codes:", duplicate_procedure_codes)

# Display the updated DataFrame
print(data['Fraud'])

# Display pairplot
sns.set(style="whitegrid")
sns.pairplot(
    data[["Patient ID", "Balance Difference", "Time Difference", "Claim ID", "Fraud"]],
    hue="Fraud",
    height=3,
    palette="Set1"
)
plt.show()



# Exclude non-numeric columns before creating the first heatmap
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
sns.heatmap(data[numeric_columns].corr(), annot=True)
plt.show()

# Perform one-hot encoding for categorical variables excluding the target variable "Fraud"
data_encoded = pd.get_dummies(data.drop(columns=['Fraud', 'Service Date', 'Claim Submission Date']))

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

# Filter the data for fraudulent cases
fraud_indices = y_test[y_test == 1].index

# Plotting the predicted values for all cases
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_pred)), y_pred, c='b', label='Predicted values', alpha=0.5)
plt.scatter(range(len(y_test)), y_test, c='r', label='Actual values', alpha=0.5)
plt.xlabel('Sample index')
plt.ylabel('Fraudulent or not (0: Non-Fraudulent, 1: Fraudulent)')
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
joblib.dump(lr, 'insurance_fraud_model.pkl')

