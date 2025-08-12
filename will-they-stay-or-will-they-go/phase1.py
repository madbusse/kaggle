import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def feature_engineering(df):
    # drop ID
    df = df.drop(columns=['crmid'])
    # Split date into separate features
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df = df.drop(columns=['date'])
    return df


def build_pipeline(cat_feats, num_feats):
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats),
        ('num', StandardScaler(), num_feats)
    ])
    # clf = XGBClassifier(
    #     n_estimators=1000,
    #     learning_rate=0.05,
    #     early_stopping_rounds=50,
    #     eval_metric='logloss',
    #     use_label_encoder=False,
    #     random_state=42
    # )
    # also try:
    clf = LGBMClassifier(objective='binary', metric='binary_logloss', random_state=42)
    return Pipeline([('pre', preprocessor), ('clf', clf)])


def main():
    parser = argparse.ArgumentParser(description='Phase 1: Hyperparameter Tuning Example')
    parser.add_argument('--train', default='train.csv')
    parser.add_argument('--test', default='test.csv')
    parser.add_argument('--output', default='submission_phase1.csv')
    args = parser.parse_args()

    # 1) Load and engineer
    train, test = load_data(args.train, args.test)
    train = feature_engineering(train)
    test  = feature_engineering(test)

    # 2) Prepare X and y
    X = train.drop(columns=['consent'])
    y = train['consent']

    # 3) Train/validation split for hyperparameter CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 4) Define feature lists
    cat_feats = ['stage', 'device1', 'device2', 'sku1', 'sku2']
    num_feats = ['length', 'crm1', 'crm2', 'crm3', 'crm4', 'year', 'month', 'day']

    # 5) Create pipeline
    pipeline = build_pipeline(cat_feats, num_feats)

    # 6) Specify hyperparameter search space
    param_dist = {
        'clf__n_estimators': [100, 200, 500],
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__max_depth': [3, 5, 7],
        'clf__subsample': [0.7, 0.8, 1.0],
        'clf__colsample_bytree': [0.7, 0.8, 1.0]
    }

    # 7) Setup RandomizedSearchCV
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=10,
        scoring='neg_log_loss',
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=42
    )

    # 8) Run hyperparameter search
    search.fit(X, y)
    print("Best hyperparameters:", search.best_params_)
    print("Best CV Log Loss:", -search.best_score_)

    # 9) Retrain best model on full training data
    best_model = search.best_estimator_
    best_model.fit(X, y)

    # 10) Predict on test
    preds = best_model.predict_proba(test)[:, 1]

    # 11) Save submission
    submission = pd.DataFrame({
        'crmid': pd.read_csv(args.test)['crmid'],
        'consent': preds
    })
    submission.to_csv(args.output, index=False)
    print(f"Submission saved to {args.output}")

if __name__ == '__main__':
    main()
