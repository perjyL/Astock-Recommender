from src.config import BUY_THRESHOLD, SELL_THRESHOLD


def make_decision(model, X_latest):
    if X_latest is None or len(X_latest) == 0:
        raise ValueError("X_latest 为空，无法预测")

    if hasattr(model, "predict_proba"):
        prob_up = float(model.predict_proba(X_latest)[0][1])
    elif hasattr(model, "predict"):
        pred = float(model.predict(X_latest)[0])
        # 兜底：如果模型输出不是概率，则按 0/1 处理
        prob_up = pred if 0.0 <= pred <= 1.0 else float(pred >= 0.5)
    else:
        raise RuntimeError("模型不支持 predict_proba / predict")
    if prob_up != prob_up:
        raise ValueError("预测概率为 NaN")

    if prob_up >= BUY_THRESHOLD:
        signal = "买入"
    elif prob_up <= SELL_THRESHOLD:
        signal = "卖出"
    else:
        signal = "持有"

    return prob_up, signal
