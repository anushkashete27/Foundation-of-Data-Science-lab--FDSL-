
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("student_exam_data.csv")

print(df.head())

plt.figure(figsize=(8, 5))
study_avg = df.groupby(df["Study Hours"].round())["Previous Exam Score"].mean()
plt.plot(study_avg.index, study_avg.values, marker="o")
plt.title("Study Hours vs Average Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Average Exam Score")
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x="Pass/Fail", y="Previous Exam Score", data=df)
plt.title("Average Exam Score by Pass/Fail")
plt.xlabel("Pass/Fail (0 = Fail, 1 = Pass)")
plt.ylabel("Previous Exam Score")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df["Previous Exam Score"], bins=10, kde=True)
plt.title("Distribution of Previous Exam Scores")
plt.xlabel("Previous Exam Score")
plt.ylabel("Number of Students")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x="Study Hours", y="Previous Exam Score", data=df)
plt.title("Study Hours vs Previous Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Previous Exam Score")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x="Pass/Fail", y="Previous Exam Score", data=df)
plt.title("Exam Score Distribution by Pass/Fail")
plt.xlabel("Pass/Fail (0 = Fail, 1 = Pass)")
plt.ylabel("Previous Exam Score")
plt.show()

result_counts = df["Pass/Fail"].value_counts().sort_index()

plt.figure(figsize=(6, 6))
plt.pie(
    result_counts,
    labels=["Fail (0)", "Pass (1)"],
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Pass/Fail Distribution")
plt.show()

plt.figure(figsize=(8, 5))

numeric_columns = [
    "Study Hours",
    "Previous Exam Score",
    "Pass/Fail"
]

correlation = df[numeric_columns].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()
