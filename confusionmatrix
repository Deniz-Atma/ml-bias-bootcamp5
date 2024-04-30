import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data
np.random.seed(42)  # Ensuring reproducibility

def generate_data(size, bias):
    """
    Generates synthetic candidate data for hiring models.
    'size' specifies the number of samples.
    'bias' is a boolean indicating if the dataset should be biased.
    """
    # Random years of experience between 1 and 10 years, normally distributed
    years_experience = np.random.normal(loc=5, scale=2, size=size).astype(int)
    # Random education levels: 1 (High School), 2 (Bachelor), 3 (Master or above)
    education_level = np.random.randint(1, 4, size=size)
    
    # Generate demographic data, where bias affects the distribution
    if bias:
        # Biased scenario: higher representation of demographic group 0 (e.g., 70%)
        demographic = np.random.choice([0, 1], size=size, p=[0.7, 0.3])
    else:
        # Balanced scenario: equal representation of demographic groups (50% each)
        demographic = np.random.choice([0, 1], size=size, p=[0.5, 0.5])
    
    # Calculate 'hire' decision based on other features plus some noise
    hire = 0.2 * years_experience + 0.3 * education_level + 0.1 * demographic + np.random.normal(0, 0.5, size=size)
    hire = (hire > np.percentile(hire, 50)).astype(int)  # Top 50% based on score get hired
    
    return pd.DataFrame({
        'Years of Experience': years_experience,
        'Education Level': education_level,
        'Demographic': demographic,
        'Hire': hire
    })

# Create both biased and balanced datasets
df_biased = generate_data(1000, True)
df_balanced = generate_data(1000, False)

def train_and_evaluate(df):
    """
    Function to train and evaluate a logistic regression model.
    'df' is the DataFrame containing the data.
    """
    # Split the data into features and target variable
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Hire', axis=1), df['Hire'], test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, cm

# Train and evaluate models on both datasets
accuracy_biased, cm_biased = train_and_evaluate(df_biased)
accuracy_balanced, cm_balanced = train_and_evaluate(df_balanced)

# Plotting results for comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.heatmap(cm_biased, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Biased Data\nAccuracy: {accuracy_biased:.2f}')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')

plt.subplot(1, 2, 2)
sns.heatmap(cm_balanced, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title(f'Balanced Data\nAccuracy: {accuracy_balanced:.2f}')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')

plt.tight_layout()
plt.show()
