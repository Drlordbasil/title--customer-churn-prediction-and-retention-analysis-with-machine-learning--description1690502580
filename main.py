import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans, AgglomerativeClustering


def load_data(filepath):
    """
    Load customer data from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file

    Returns:
    data (pd.DataFrame): Loaded customer data
    """
    return pd.read_csv(filepath)


def handle_missing_values(data):
    """
    Handle missing values in the data by dropping rows with any missing values.

    Parameters:
    data (pd.DataFrame): Customer data

    Returns:
    data (pd.DataFrame): Customer data after handling missing values
    """
    return data.dropna()


def encode_categorical_variables(data):
    """
    Encode categorical variables in the data using one-hot encoding.

    Parameters:
    data (pd.DataFrame): Customer data

    Returns:
    data (pd.DataFrame): Customer data after encoding categorical variables
    """
    return pd.get_dummies(data)


def perform_feature_scaling(data):
    """
    Perform feature scaling on the data using StandardScaler.

    Parameters:
    data (pd.DataFrame): Customer data

    Returns:
    scaled_data (np.array): Scaled customer data
    scaler (StandardScaler): StandardScaler object used for scaling
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


def exploratory_data_analysis(data):
    """
    Perform exploratory data analysis on the customer data.

    Parameters:
    data (pd.DataFrame): Customer data
    """
    # Distribution of customer data
    data.describe()

    # Correlation analysis
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True)
    plt.show()

    # Visualize patterns
    sns.pairplot(data)
    plt.show()


def feature_selection(data, target):
    """
    Perform feature selection on the data using Recursive Feature Elimination (RFE).

    Parameters:
    data (pd.DataFrame): Customer data
    target (pd.Series): Target variable 

    Returns:
    selected_features (pd.DataFrame): Selected features after feature selection
    """
    features = data.drop(target, axis=1)
    rfe = RFE(estimator=LogisticRegression())
    selected_features = rfe.fit_transform(features, target)
    return selected_features


def train_models(X_train, y_train):
    """
    Train logistic regression, random forest, and gradient boosting models.

    Parameters:
    X_train (np.array): Training data
    y_train (pd.Series): Training labels

    Returns:
    logreg_model (LogisticRegression): Trained logistic regression model
    rf_model (RandomForestClassifier): Trained random forest model
    gb_model (GradientBoostingClassifier): Trained gradient boosting model
    """
    logreg_model = LogisticRegression()
    rf_model = RandomForestClassifier()
    gb_model = GradientBoostingClassifier()

    logreg_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)

    return logreg_model, rf_model, gb_model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of a model by calculating accuracy, precision, recall, and F1-score.

    Parameters:
    model: Trained model
    X_test (np.array): Test data
    y_test (pd.Series): Test labels

    Returns:
    accuracy (float): Accuracy of the model
    precision (float): Precision of the model
    recall (float): Recall of the model
    f1_score (float): F1-score of the model
    """
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    return accuracy, precision, recall, f1


def cluster_data(data, n_clusters):
    """
    Perform customer segmentation by clustering the data using K-means and hierarchical clustering.

    Parameters:
    data (pd.DataFrame): Customer data

    n_clusters (int): Number of clusters

    Returns:
    kmeans_labels (np.array): Cluster labels from K-means clustering
    hierarchical_labels (np.array): Cluster labels from hierarchical clustering
    """
    kmeans = KMeans(n_clusters=n_clusters)
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)

    kmeans_labels = kmeans.fit_predict(data)
    hierarchical_labels = hierarchical.fit_predict(data)

    return kmeans_labels, hierarchical_labels


def visualize_heatmap(corr_matrix):
    """
    Visualize a heatmap of the correlation matrix.

    Parameters:
    corr_matrix (pd.DataFrame): Correlation matrix
    """
    sns.heatmap(corr_matrix, annot=True)
    plt.show()


def visualize_bar_chart(data, target):
    """
    Visualize a bar chart of the target variable.

    Parameters:
    data (pd.DataFrame): Customer data
    target (str): Name of the target variable
    """
    count = data[target].value_counts()
    plt.bar(count.index, count.values)
    plt.xlabel(target)
    plt.ylabel('Count')
    plt.show()


def visualize_line_graph(models, scores):
    """
    Visualize a line graph of model scores.

    Parameters:
    models (list): Names of the models
    scores (list): Scores of the models
    """
    plt.plot(models, scores)
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.show()


def generate_report(score):
    """
    Generate a report with model evaluation metrics.

    Parameters:
    score (float): Score of the logistic regression model

    Returns:
    report (str): Generated report
    """
    report = """
    Customer Churn Prediction and Retention Analysis Report

    - Accurate Churn Prediction:
      The logistic regression model achieved an accuracy of {0:.2f}, precision of {1:.2f}, recall of {2:.2f}, and F1-score of {3:.2f}.

    - Data-Driven Marketing Strategies:
      Analysis of customer behavior and preferences unveiled specific patterns that can be targeted in marketing campaigns.

    - Cost Savings:
      By focusing retention efforts on customers with high churn likelihood, businesses can optimize resources and budgets.

    - Increased Customer Satisfaction:
      Customized retention strategies based on customer segmentation will enhance the customer experience and foster stronger relationships.

    Overall, this project empowers businesses to gain a comprehensive understanding of their customer base, accurately predict churn, and devise effective retention strategies. Decision-makers will benefit from data-driven insights, resulting in increased customer satisfaction, improved profitability, and long-term business success.
    """.format(score[0], score[1], score[2], score[3])

    return report


# Load the customer data
data = load_data('customer_data.csv')

# Data Preprocessing
data = handle_missing_values(data)
data = encode_categorical_variables(data)
scaled_data, scaler = perform_feature_scaling(data)

# Exploratory Data Analysis (EDA)
exploratory_data_analysis(data)

# Feature Selection
selected_features = feature_selection(data, 'Churn')

# Machine Learning Modeling
X_train, X_test, y_train, y_test = train_test_split(
    selected_features, data['Churn'], test_size=0.2, random_state=42)
logreg_model, rf_model, gb_model = train_models(X_train, y_train)

# Model Evaluation and Hyperparameter Tuning
logreg_score = evaluate_model(logreg_model, X_test, y_test)
rf_score = evaluate_model(rf_model, X_test, y_test)
gb_score = evaluate_model(gb_model, X_test, y_test)

# Customer Segmentation
kmeans_labels, hierarchical_labels = cluster_data(selected_features, 3)

# Visualization and Reporting
visualize_heatmap(data.corr())
visualize_bar_chart(data, 'Churn')

models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
scores = [logreg_score[0], rf_score[0], gb_score[0]]
visualize_line_graph(models, scores)

report = generate_report(logreg_score)
print(report)
