from sklearn.ensemble import RandomForestClassifier
from src.config import RANDOM_STATE


def train_model(X, y):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=RANDOM_STATE
    )
    model.fit(X, y)
    return model
