import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

# 1) Load
df = pd.read_csv("train.csv")

# 2) Split off a local “test” set
train_df, val_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df["consent"]
)

# 3) Define features & preprocessing just like your submission pipeline
features = ["length", "stage", "device1", "device2", "date", "sku1", "sku2", "crm1", "crm2", "crm3", "crm4"]
cat_feats = ["stage", "device1", "device2", "sku1", "sku2"]
num_feats = ["length", "crm1", "crm2", "crm3", "crm4"]

preprocessor = ColumnTransformer([
    ("cats", OneHotEncoder(handle_unknown="ignore"), cat_feats),
    ("nums", StandardScaler(), num_feats),
], remainder="drop")

pipeline = Pipeline([
    ("pre", preprocessor),
    ("clf", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )),
])

# 4) Fit on training fold
pipeline.fit(train_df[features], train_df["consent"])

# 5) Predict and compute local Log Loss
val_pred = pipeline.predict_proba(val_df[features])[:, 1]
score = log_loss(val_df["consent"], val_pred)
print(f"Local validation Log Loss: {score:.5f}")
