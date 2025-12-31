from src.config import BUY_THRESHOLD, SELL_THRESHOLD


def make_decision(model, X_latest):
    prob_up = model.predict_proba(X_latest)[0][1]

    if prob_up >= BUY_THRESHOLD:
        signal = "买入"
    elif prob_up <= SELL_THRESHOLD:
        signal = "卖出"
    else:
        signal = "持有"

    return prob_up, signal
