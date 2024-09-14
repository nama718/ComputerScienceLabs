import pandas as panda
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = panda.read_csv("insurance.csv")

# Convert categorical variables to category type
data['sex'] = data['sex'].astype('category')
data['region'] = data['region'].astype('category')
data['smoker'] = data['smoker'].astype('category')

# Display pairplot
sns.set(style="whitegrid")
sns.pairplot(
    data[["age", "bmi", "charges", "smoker"]],
    hue="smoker",
    height=3,
    palette="Set1"
)
plt.show()

# Exclude non-numeric columns before creating the first heatmap
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
sns.heatmap(data[numeric_columns].corr(), annot=True)
plt.show()

# Perform one-hot encoding
data = panda.get_dummies(data)

# Display the second heatmap
sns.heatmap(data.corr(), annot=True)
plt.show()

# Display the columns after one-hot encoding
print(data.columns)

# creating the Regression Model

y = data["charges"]
X = data.drop("charges", axis=1)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=0.80, test_size=.20,
    random_state=1)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test).round(3)
print("R-squared score on the test set:", lr.score(X_test, y_test).round(3))
print("R-squared score on the train set:", lr.score(X_train, y_train).round(3))
y_pred = lr.predict(X_test)
print("Root Mean Squared Error on the test set:", math.sqrt(mean_squared_error(y_test, y_pred)))


#Predicting Model
data_new = X_train[:1]
print(lr.predict(data_new))
print(y_train[:1])


# Displaying Scatter Plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs. Predicted Charges")
plt.show()


# Displaying Residual Plot
residuals = y_test - y_pred
plt.scatter(X_test['age'], residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Age")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

