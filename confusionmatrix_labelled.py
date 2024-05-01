import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Setting a seed for reproducibility ensures that the random numbers generated are the same every time the script runs.
np.random.seed(42)

def generate_data(size, bias):
    """
    Function to generate synthetic hiring data.
    'size' specifies the number of samples to generate.
    'bias' determines whether the data should be biased towards a demographic.
    """
    # Generating random 'years of experience' data, normally distributed around 5 years with a standard deviation of 2.
    years_experience = np.random.normal(loc=5, scale=2, size=size).astype(int)
    # Generating random 'education level' data, with values ranging from 1 (High School) to 3 (Master's or higher).
    education_level = np.random.randint(1, 4, size=size)
    # Generating 'demographic' data, either biased (70-30 split) or balanced (50-50 split).
    if bias:
        demographic = np.random.choice([0, 1], size=size, p=[0.7, 0.3])
    else:
        demographic = np.random.choice([0, 1], size=size, p=[0.5, 0.5])
    
    # Combining features to determine hiring decisions, with some random noise to simulate real-world unpredictability.
    hire = 0.2 * years_experience + 0.3 * education_level + 0.1 * demographic + np.random.normal(0, 0.5, size=size)
    # Deciding on hiring based on whether the score is in the top 50%.
    hire = (hire > np.percentile(hire, 50)).astype(int)
    
    # Returning the data as a DataFrame for easier manipulation and analysis.
    return pd.DataFrame({
        'Years of Experience': years_experience,
        'Education Level': education_level,
        'Demographic': demographic,
        'Hire': hire
    })

# Generating both biased and balanced datasets with 1000 samples each.
df_biased = generate_data(1000, True)
df_balanced = generate_data(1000, False)

def train_and_evaluate(df):
    """
    Function to train and evaluate a logistic regression model.
    'df' is the DataFrame containing the hiring data.
    """
    # Splitting the data into training and testing sets (80% train, 20% test).
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Hire', axis=1), df['Hire'], test_size=0.2, random_state=42)
    # Creating and training the logistic regression model.
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    # Making predictions on the testing set.
    y_pred = model.predict(X_test)
    # Calculating the accuracy of the model.
    accuracy = accuracy_score(y_test, y_pred)
    # Generating a confusion matrix to visualize model performance.
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, cm

# Training and evaluating models on both datasets.
accuracy_biased, cm_biased = train_and_evaluate(df_biased)
accuracy_balanced, cm_balanced = train_and_evaluate(df_balanced)

# Plotting the confusion matrices for both scenarios with intuitive labels and formatting.
plt.figure(figsize=(12, 6))
sns.set(font_scale=1.2)  # Adjust font scale for better readability

# List of titles for each subplot.
titles = ['Biased Data', 'Balanced Data']
# List of confusion matrices from each model.
cms = [cm_biased, cm_balanced]
# List of accuracies from each model.
accuracies = [accuracy_biased, accuracy_balanced]

for i, cm in enumerate(cms):
    plt.subplot(1, 2, i+1)
    # Labels for each part of the confusion matrix.
    labels = ['Correct Rejection', 'Wrong Hire', 'Missed Hire', 'Correct Hire']
    # Count of each outcome.
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    # Percentage of each outcome.
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    # Combining labels, counts, and percentages for clear annotation.
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(labels, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    # Plotting the heatmap.
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues' if i == 0 else 'Greens', cbar=False)
    plt.title(f'{titles[i]}\nAccuracy: {accuracies[i]:.2f}')
    plt.xlabel('Predicted Category')
    plt.ylabel('Actual Category')

plt.tight_layout()
plt.show()
