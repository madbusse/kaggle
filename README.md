# Kaggle Projects

## Will They Stay or Will They Go?
[Link to challenge HERE](https://www.kaggle.com/competitions/will-they-stay-or-will-they-go/leaderboard)

In this competition, I worked with an anonymized dataset to build a binary classification model that predicts whether a user will consent to data sharing based on historical behavioral and CRM features (including device information).

### Stage 1
I used sklearn for hyperparameter tuning to generate a prediction between 0 and 1 whether a person would consent to data collection (1) or not (0). Evaluated on a log loss scale. Current best score = 0.55855.

### Stage 2
The task is to rank users by their potential revenue — simulating a real-world targeting scenario with misclassification penalties.
