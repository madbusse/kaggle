import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
print("import success")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_y = train['consent']
# we drop consent bc it's training data and crmid bc it has no meaning
train_X = train.drop(columns=['consent', 'crmid'])

train_X, validate_X, train_y, validate_y = train_test_split(train_X, train_y, test_size=0.25, random_state=1)
print("data loaded")

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(11,)),
        tf.keras.layers.Dense(60, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("model built")
    return model
    
def evaluate_model():
    estimator = KerasClassifier(model=build_model, epochs=3, batch_size=5, verbose=0)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    results = cross_val_score(estimator, train_X, train_y, cv=kfold, scoring="neg_log_loss")
    print("Baseline estimated accuracy mean (SD): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def main():
    evaluate_model()

if __name__ == "__main__":
    main()