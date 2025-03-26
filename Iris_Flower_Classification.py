import logging
from google.colab import files
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import shap

# Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def upload_and_load_data() -> pd.DataFrame:
    """
    Uploads a CSV file using Google Colab's file uploader and loads it into a Pandas DataFrame.

    Returns:
        DataFrame: Loaded data.
    """
    try:
        uploaded = files.upload()
        file_name = list(uploaded.keys())[0]    # Get the uploaded file name
        logging.info(f"File '{file_name}' uploaded successfully.")
        data = pd.read_csv(file_name)
        return data
    except Exception as e:
        logging.error("Error during file upload or loading.", exc_info=True)
        raise e


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the data: standardizes column names, handles missing values,
    removes duplicates, converts categorical labels, and scales features.

    Args:
        data (DataFrame): The raw input data.

    Returns:
        DataFrame: Preprocessed and scaled data.
    """
    # Standardize column names (lowercase, strip spaces)
    data.columns = data.columns.str.lower().str.strip()

    # Check for missing values
    logging.info("Data info:")
    data.info()
    logging.info(f"Missing values in dataset:\n{data.isnull().sum()}")

    # Remove duplicate rows
    data.drop_duplicates(inplace=True)

    # Drop any unnecessary columns (if required)
    if 'id' in data.columns:
        data.drop(columns=['id'], inplace=True)

    # Convert categorical labels to numeric values
    label_encoder = LabelEncoder()
    if 'species' in data.columns:
        data['species'] = label_encoder.fit_transform(data['species'])

    # Feature Scaling
    scaler = StandardScaler()
    X = data.drop(columns=['species'])
    data_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    data_scaled['species'] = data['species']

    # Handle any potential NaN values in 'species'
    data_scaled.dropna(subset=['species'], inplace=True)

    logging.info("Preprocessing completed.")
    return data_scaled


def perform_eda(data: pd.DataFrame):
    """
    Performs enhanced exploratory data analysis (EDA) by generating a pairplot,
    correlation heatmap, and class distribution countplot.

    Args:
        data (DataFrame): Preprocessed data.
    """
    # Pairplot
    sns.pairplot(data, hue='species', palette='husl')
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # Class Distribution Countplot
    plt.figure(figsize=(6, 4))
    sns.countplot(x='species', data=data, palette='husl')
    plt.title("Class Distribution")
    plt.show()

    logging.info("EDA completed.")


def split_data(data: pd.DataFrame):
    """
    Splits the data into training and testing sets.

    Args:
        data (DataFrame): Preprocessed data.

    Returns:
        X_train, X_test, y_train, y_test: Splitted datasets.
    """
    X = data.drop(columns=['species'])
    y = data['species']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train) -> RandomForestClassifier:
    """
    Trains a Random Forest model using GridSearchCV for hyperparameter tuning.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        best_model: The best Random Forest estimator from grid search.
    """
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 4, 6]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    logging.info(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def plot_feature_importance(model: RandomForestClassifier, X):
    """
    Plots feature importance from the Random Forest model.

    Args:
        model: Trained Random Forest model.
        X: Feature set used in training.
    """
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=feature_importance_df['Importance'], y=feature_importance_df['Feature'], palette='viridis')
    plt.title("Feature Importance in Random Forest Model")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.show()


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model by printing accuracy, classification report, and plotting the confusion matrix.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model Accuracy: {accuracy}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def explain_with_shap(model, X_test):
    """
    Uses SHAP to explain the model predictions.

    Args:
        model: Trained model.
        X_test: Test features.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")


def main():
    # Step 1: Upload and load data
    data = upload_and_load_data()

    # Step 2: Preprocess data
    data_preprocessed = preprocess_data(data)

    # Step 3: Perform EDA
    perform_eda(data_preprocessed)

    # Step 4: Split Data
    X_train, X_test, y_train, y_test = split_data(data_preprocessed)

    # Step 5: Train model with hyperparameter tuning
    best_model = train_model(X_train, y_train)

    # Plot Feature Importance
    plot_feature_importance(best_model, X_train)

    # Step 6: Evaluate model
    evaluate_model(best_model, X_test, y_test)

    # Step 7: Model explainability using SHAP
    explain_with_shap(best_model, X_test)

    # Step 8: Save the best model
    joblib.dump(best_model, 'iris_model.pkl')
    logging.info("Model saved as 'iris_model.pkl'")


if __name__ == '__main__':
    main()


# Haa thikk haa... Chatgpt se batchit krke bnaya ha