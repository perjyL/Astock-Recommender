import akshare as ak
import pandas as pd

from src.data_loader import get_stock_history
from src.feature_engineering import add_features
from src.model import train_model


def get_recommendation(prob):
    if prob >= 0.6:
        return "Buy"
    elif prob >= 0.4:
        return "Hold"
    else:
        return "Sell"


def hs300_recommendation():
    # 获取沪深300成分股
    hs300 = ak.index_stock_cons_csindex(symbol="000300")

    results = []

    for _, row in hs300.iterrows():
        code = row["成分券代码"]
        name = row["成分券名称"]

        try:
            # 1️⃣ 历史数据
            df = get_stock_history(code)

            # 2️⃣ 特征工程
            df = add_features(df)

            features = [
                "MA5", "MA10", "MA20",
                "DIF", "DEA", "MACD",
                "VOL_MA5", "Volatility"
            ]

            X = df[features]
            y = df["Target"]

            # 3️⃣ 训练模型
            model = train_model(X[:-1], y[:-1])

            # 4️⃣ 预测最后一天
            prob = model.predict_proba(X.iloc[[-1]])[0][1]
            rec = get_recommendation(prob)

            results.append({
                "Code": code,
                "Name": name,
                "Up_Prob": round(prob, 4),
                "Recommendation": rec
            })

            print(f"{code} {name} → {rec} ({prob:.2f})")

        except Exception as e:
            print(f"{code} {name} 数据异常，跳过")
            continue

    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values("Up_Prob", ascending=False)
    df_result.insert(0, "Rank", range(1, len(df_result) + 1))

    return df_result
