import pandas as pd
import numpy as np
from faker import Faker
fake = Faker()

# Define the size of the dataset
num_records = 1000

# Generate synthetic data for healthcare claims with additional details
np.random.seed(42)  # For reproducibility
data = {
    "Claim ID": [fake.unique.random_int(min=100000, max=999999) for _ in range(num_records)],
    "Patient Risk Category": np.random.choice(['Low', 'Medium', 'High'], num_records),
    "Procedure Type": np.random.choice(['Conservative', 'Diagnostic', 'Invasive'], num_records, p=[0.5, 0.3, 0.2]),
    "Number of Procedures": [np.random.randint(1, 3) if risk != 'High' else np.random.randint(1, 5) for risk in np.random.choice(['Low', 'Medium', 'High'], num_records)],
    "Cost": [np.random.uniform(200, 2000) if procedure != 'Invasive' else np.random.uniform(2000, 10000) for procedure in np.random.choice(['Conservative', 'Diagnostic', 'Invasive'], num_records, p=[0.5, 0.3, 0.2])]
}

df = pd.DataFrame(data)

# Export DataFrame to an Excel file
df.to_excel("healthcare_claims_data.xlsx", index=False)
# Print a message to indicate completion
print("Dataset generated and saved to 'InsuranceFraud.xlsx'.")