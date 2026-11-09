
import pandas as pd
import pickle

# Read the CSV file
df = pd.read_csv("ts_ecet_new_dataset_2022_2026_validated.csv")

# Display column data types
print("DATA TYPES:")
print(df.dtypes)

# Remove duplicate records
df = df.drop_duplicates()

# Import train-test split
from sklearn.model_selection import train_test_split

# Separate input and output
x = df[[
    "Year",
    "Diploma_Branch",
    "Mathematics_Marks",
    "Physics_Marks",
    "Chemistry_Marks",
    "Engineering_Marks"
]]

# Convert branch names into numerical values
x = pd.get_dummies(x, columns=["Diploma_Branch"])

# Integrated Rank model
y = df["Integrated_Rank"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LinearRegression

integrated_model = LinearRegression()

# Train integrated rank model
integrated_model.fit(x_train, y_train)

# Predict integrated ranks
y_pred = integrated_model.predict(x_test)

print("\nINTEGRATED RANK PREDICTIONS:")
print(y_pred[:10])

print("\nACTUAL INTEGRATED RANKS:")
print(y_test.head(10))

# Save integrated rank model
pickle.dump(
    (integrated_model, x.columns),
    open("integrated_rank_model.pkl", "wb")
)


# Branch Rank model
y = df["Branch_Rank"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

branch_model = LinearRegression()

# Train branch rank model
branch_model.fit(x_train, y_train)

# Predict branch ranks
y_pred = branch_model.predict(x_test)

print("\nBRANCH RANK PREDICTIONS:")
print(y_pred[:10])

print("\nACTUAL BRANCH RANKS:")
print(y_test.head(10))

# Save branch rank model
pickle.dump(
    (branch_model, x.columns),
    open("branch_rank_model.pkl", "wb")
)

print("\nBOTH MODELS TRAINED SUCCESSFULLY!")
