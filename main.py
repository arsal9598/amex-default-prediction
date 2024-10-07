import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, categorical_cols):
    # Drop rows where 50% or more of the columns have NaN values
    df = df.dropna(thresh=len(df.columns) * 0.5)

    # Convert datetime columns to numeric if any exist
    for col in df.select_dtypes(include=['datetime64']):
        df[col] = df[col].astype('int64') // 10**9  # Convert to seconds since epoch

    # Encode categorical columns
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    return df

def main():
    # Load the training data
    train_df = pd.read_feather("feather-dataset/train_data_f32.ftr")
    print("Original training data shape:", train_df.shape)

    # Preprocess training data
    train_df = preprocess_data(train_df, categorical_cols)

    # Sample 1% of the data
    train_df_sampled = train_df.sample(frac=0.5, random_state=12)
    print("Sampled training data shape:", train_df_sampled.shape)

    # Check if target column exists
    target_col = 'target'  # Replace with your actual target column name
    if target_col not in train_df_sampled.columns:
        print(f"Error: Target column '{target_col}' not found in the DataFrame.")
        return

    # Split data into features and target variable
    X = train_df_sampled.drop(columns=[target_col])
    y = train_df_sampled[target_col]

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=12)

    # Initialize LightGBM classifier
    lgb_model = LGBMClassifier(n_estimators=1000, learning_rate=0.1, max_depth=-1)

    # Fit the model
    lgb_model.fit(X_train, y_train)

    # Load and preprocess the test data
    test_df = pd.read_feather("feather-dataset/test_data_f32.ftr")
    print("Original test data shape:", test_df.shape)

    # Drop columns not in training data
    test_df = test_df[X.columns]
    
    # Preprocess test data
    test_df = preprocess_data(test_df, categorical_cols)
    
    # Make predictions on test data
    predictions = lgb_model.predict_proba(test_df)[:, 1]  # Get probability of the positive class

    # Create submission DataFrame
    submission = pd.DataFrame({
        'customer_ID': test_df['customer_ID'],  # Adjust this to your actual customer ID column name
        'prediction': predictions
    })

    # Save submission file
    submission.to_csv('submissions.csv', index=False)
    print("Submission file created:", submission.shape)

if __name__ == "__main__":
    categorical_cols = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    main()