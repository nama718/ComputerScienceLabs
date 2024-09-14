import pandas as pd
import numpy as np
from faker import Faker

fake = Faker()

# Define the size of the dataset
num_records = 100


# Generate synthetic data for healthcare claims
def generate_data(num_records):
    data = {
        "Claim ID": [fake.unique.random_int(min=100000, max=999999) for _ in range(num_records)],
        "Patient ID": [fake.unique.random_int(min=1000, max=9999) for _ in range(num_records)],
        "Provider ID": [fake.unique.random_int(min=10000, max=99999) for _ in range(num_records)],
        "Service Date": [fake.date_between(start_date="-3y", end_date="today") for _ in range(num_records)],
        "Diagnosis Code": [fake.bothify(text='?##.##', letters='ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(num_records)],
        "Procedure Code": [fake.bothify(text='####', letters='ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(num_records)],
        "Billing Amount": [round(np.random.uniform(100, 10000), 2) for _ in range(num_records)],
        "Paid Amount": [round(np.random.uniform(50, 5000), 2) for _ in range(num_records)],
        "Patient Responsibility": [round(np.random.uniform(10, 500), 2) for _ in range(num_records)],
    }
    df = pd.DataFrame(data)

    # Introduce irregular billing pattern for a random provider
    irregular_provider_id = np.random.choice(df['Provider ID'].unique())
    irregular_indices = df[df['Provider ID'] == irregular_provider_id].index
    df.loc[irregular_indices, 'Billing Amount'] *= np.random.uniform(1.5, 2.0, len(irregular_indices))

    # Add a new column named 'Balance Difference'
    df['Balance Difference'] = df['Billing Amount'] - df['Paid Amount'] - df['Patient Responsibility']

    # Generate Claim Submission Date based on Service Date
    df["Claim Submission Date"] = [fake.date_between(start_date=date, end_date="+60d") for date in
                                   df["Service Date"]]

    return df


# Generate the dataset
df = generate_data(num_records)

# Basic analysis example (find top 5 providers by average billing amount)
print(df.groupby('Provider ID')['Billing Amount'].mean().sort_values(ascending=False).head(5))

# Save the dataset to an Excel file
df.to_excel('InsuranceFraud.xlsx', index=False)

# Print a message to indicate completion
print("Dataset generated and saved to 'InsuranceFraud.xlsx'.")
