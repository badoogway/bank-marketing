import click
import pickle
import pandas as pd

from preprocessing import DataSplitter, DataTransformer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

@click.command()
@click.option('--csv_path', required=True, help='Path to CSV')
@click.option('--csv_separator', required=True, help='Separator for CSV file')
@click.option('--model_output_path', required=True, help='Path to save the trained model')
def train(csv_path, csv_separator, model_output_path):

    # Define target column and feature lists
    target_column = 'y'
    num_features = ['age', 'balance', 'campaign', 'pdays', 'previous']
    cat_features = [
        'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome',
    ]
    
    # Initialize data splitter and data transformer
    splitter = DataSplitter()
    transformer = DataTransformer(
        num_features=num_features, 
        cat_features=cat_features + ['active_contact', 'without_debt']
    )
    
    # Load data from CSV file
    click.echo("[LOG] Loading data..")
    df = pd.read_csv(csv_path, sep=csv_separator)

    # Split data into training and validation sets
    click.echo("[LOG] Dividing data into training and validation sets...")
    df_train, df_valid = splitter.split_data(
        df, 
        num_features=num_features, cat_features=cat_features, target_column=target_column
    )

    # Generate additional features for training and validation sets
    click.echo("[LOG] Generating features for training and validation sets...")
    df_train = transformer.add_custom_features(df_train)
    df_valid = transformer.add_custom_features(df_valid)

    # Transform datasets into trainable format
    click.echo("[LOG] Transforming datasets to trainable format...")
    X_train = transformer.fit_transform(df_train)
    X_valid = transformer.transform(df_valid)

    # Prepare target labels
    y_train = (df_train[target_column] == 'yes').astype(int).values
    y_valid = (df_valid[target_column] == 'yes').astype(int).values

    # Train XGBoost model on the dataset
    click.echo("[LOG] Training XGBoost on the train dataset...")
    model = XGBClassifier(
        n_estimators=267, 
        max_depth=1, 
        learning_rate=0.41850597186514554, 
        min_child_weight=8
    ) # parameters from fine-tuning, check out brief-exploration.ipynb in "notebooks" folder
    model.fit(X_train, y_train)

    # Evaluate the model with AUC-ROC score on the validation set
    score = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
    click.echo(f'[INFO] AUC-ROC score for the validation set: {round(score, 4)}')

    # Save the trained model and data transformer
    click.echo(f"[LOG] Saving model...")
    with open(model_output_path, 'wb') as f:
        pickle.dump((model, transformer), f)

    click.echo(f"[LOG] The model successfully saved in '{model_output_path}'")


if __name__ == '__main__':
    train()
