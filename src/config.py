# 指数代码（沪深300）
# INDEX_CODE = "000300"
INDEX_CODE = "000016"
# ==================================================

# 回测与训练时间范围
START_DATE = "20230101"
END_DATE = "20260112"
# ==================================================

# 投资决策阈值
BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45
# ==================================================

# 随机种子
RANDOM_STATE = 42
# ==================================================


# 模型选择
# 1.回归模型
MODEL_TYPE_REG = "xgb_reg"     # ridge / rf_reg / xgb_reg

# 2.分类模型
# MODEL_TYPE_CLS = "transformer"
MODEL_TYPE_CLS = "randomforest"
# MODEL_TYPE_CLS = "xgboost"


# 回归目标
# =========================
RETURN_TARGET = "ret_5d"   # ret_5d / ret_10d


# 组合构建参数
# =========================
TOP_K = 20                # 多头持仓数量
MIN_TRAIN_SIZE = 200
PORTFOLIO_TARGET_COL = "ret_5d"
REALIZED_RET_COL = "ret_1d_fwd"
REBALANCE_N = 5              # 1=日调仓；5=5日调仓
COST_RATE = 0.001            # 千分之一单边（示例）
# ==================================================






# Transformer 参数
TRANSFORMER_WINDOW = 20
TRANSFORMER_EPOCHS = 8
USE_JOINT_TRANSFORMER = True    # 是否进行联合训练

# 联合模型微调（逐股）
USE_JOINT_FINETUNE = True
JOINT_FINETUNE_EPOCHS = 1
JOINT_FINETUNE_LR = 1e-4
# ==================================================
