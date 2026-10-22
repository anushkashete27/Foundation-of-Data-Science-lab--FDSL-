import pandas as pd
from ydata_profiling import ProfileReport

# Step 1: Load dataset
df = pd.read_csv("AI Job Market & Salary Trends 2025.csv")

# Step 2: Display basic information
print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Step 3: Generate Pandas Profiling Report
profile = ProfileReport(
    df,
    title="Exploratory Data Analysis Report",
    explorative=True
)

# Step 4: Save report as HTML
profile.to_file("EDA_Report.html")

print("\nEDA Report generated successfully!")