# Ashares-Recommender

面向 A 股指数成分股的量化推荐/回测小项目：用 AkShare 拉历史行情，做技术指标特征，训练模型预测下一交易日上涨概率或未来收益，并输出 Buy/Hold/Sell 与 Top-K 组合回测结果。

## 特性
- 拉取指数成分股与个股日线行情（前复权）
- 构建均线、MACD、成交量均线、收益率与波动率等特征
- 支持分类模型（随机森林 / XGBoost / Transformer）与回归模型（Ridge / RF / XGB）
- 批量生成推荐结果并输出 CSV
- 支持逐日 Walk-Forward 回测与 Top-K 组合分桶滚动回测
- 提供价格/均线、MACD、成交量可视化与回测图表输出

## 项目结构
```bash
Ashares-Recommender/
├─ Stock_Recommender.py            # 主入口脚本，默认批量生成指数成分股推荐结果
├─ Example_DataDisplay.py          # 数据与指标展示示例脚本
├─ Stock_Recommender_Backtest.py   # 逐日 Walk-Forward 回测（分类）
├─ Stock_Recommender_Backtest_1.py # Top-K 组合分桶滚动回测（回归）
├─ Stock_Recommender_ActionGuide.py # 明日操作行动指南（基于 Top-K 回归策略）
├─ README.md
├─ requirements.txt
├─ output/
│  └─ hs300_recommendation.csv     # 示例输出文件
└─ src/
   ├─ config.py                    # 全局配置（指数、日期范围、阈值、模型类型等）
   ├─ data_loader.py               # 数据拉取（成分股列表、个股行情、指数行情）
   ├─ feature_engineering.py       # 技术指标与标签构建
   ├─ model.py                     # 模型训练（分类/回归、联合 Transformer）
   ├─ predictor.py                 # 概率到交易信号映射
   ├─ recommender.py               # 指数成分股批量推荐与排序
   ├─ backtest.py                  # 单股/全指数分类回测
   ├─ backtest_portfolio.py        # Top-K 组合分桶滚动回测
   └─ visualization.py             # 绘图工具
```

## 快速开始
1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 运行推荐脚本

```bash
python Stock_Recommender.py
```

运行后会在 `output/hs300_recommendation.csv` 生成结果，并打印 Top 10。

3. 回测脚本（可选）

```bash
# 分类模型：逐日 Walk-Forward 回测
python Stock_Recommender_Backtest.py

# 回归模型：Top-K 组合分桶滚动回测
python Stock_Recommender_Backtest_1.py
```

4. 明日操作行动指南（可选）

```bash
python Stock_Recommender_ActionGuide.py --capital 1000000
```

## 使用方式
- 批量推荐（默认入口）：
  - `Stock_Recommender.py` 调用 `recommender.hs300_recommendation()`
- 单只股票示例：
  - `Stock_Recommender.py` 中的 `main()`
  - 注意：`main()` 使用 `predictor.make_decision`（依赖 `predict_proba`），适用于随机森林 / XGBoost；若 `MODEL_TYPE_CLS = "transformer"`，需使用 Transformer 预测逻辑
- 回测：
  - `Stock_Recommender_Backtest.py`：逐日 Walk-Forward（分类）
  - `Stock_Recommender_Backtest_1.py`：Top-K 组合分桶滚动（回归）
- 明日行动指南：
  - `Stock_Recommender_ActionGuide.py`：根据最近收盘数据生成次日建仓清单与资金分配

## 配置
位置：`src/config.py`
- `INDEX_CODE`：指数代码（默认 `000016`）
- `START_DATE` / `END_DATE`：训练与数据拉取范围
- `FEATURE_COLS`：特征列（与 `feature_engineering.add_features` 对齐）
- `BUY_THRESHOLD` / `SELL_THRESHOLD`：推荐阈值
- `MODEL_TYPE_CLS`：分类模型类型（`randomforest` / `xgboost` / `transformer`）
- `MODEL_TYPE_REG`：回归模型类型（`ridge` / `rf_reg` / `xgb_reg`）
- `TRANSFORMER_WINDOW` / `TRANSFORMER_EPOCHS`：Transformer 训练参数
- `USE_JOINT_TRANSFORMER`：是否启用联合训练
- `USE_JOINT_FINETUNE`：是否对联合模型做逐股微调
- `JOINT_FINETUNE_EPOCHS` / `JOINT_FINETUNE_LR`：逐股微调参数
- `TOP_K` / `HOLD_N` / `MIN_TRAIN_SIZE`：组合回测核心参数
- `PORTFOLIO_TARGET_COL` / `REALIZED_RET_COL`：组合回测目标与真实收益列
- `WEIGHT_MODE` / `SOFTMAX_TAU` / `COST_RATE`：组合权重与成本假设
- `BACKTEST_START` / `BACKTEST_END`：组合回测区间
- `FIG_DIR`：回测图表输出目录
- `PLOT_FONT`：绘图字体方案（`auto` / `simhei` / `pingfang` / `en`）
- `PLOT_LANG`：图表文字语言（`zh` / `en`）

## 输出
`output/hs300_recommendation.csv` 字段说明：
- `Rank`：排序名次
- `Code`：股票代码
- `Name`：股票名称
- `Up_Prob`：预测上涨概率
- `Recommendation`：Buy / Hold / Sell

回测输出：
- `output/hs300_backtest_results.csv`：分类 Walk-Forward 回测逐日记录
- `output/portfolio_rollover_daily_details.csv`：Top-K 分桶回测明细
- `output/figs/`：净值曲线、回撤、收益分布、预测散点等图表

明日行动指南输出：
- `output/next_day_action_guide.csv`：次日 Top-K 建仓清单与建议资金

## 依赖
- `requirements.txt`：akshare、pandas、numpy、scikit-learn、xgboost、matplotlib、ta
- 若启用 Transformer：需额外安装 `torch`
- `tqdm` 用于回测进度条，未安装会自动降级为普通循环

## 注意
- AkShare 需要网络访问；个别股票数据异常会被跳过
- Transformer 对样本长度有要求，样本过短会报错
- 可视化字体默认按系统自动选择（Windows/ macOS 会优先选择常见中文字体）
  - 也可在 `PLOT_FONT` / `PLOT_LANG` 中切换到 PingFang 或英文标签以避免字体警告

## 算法简述
先用历史行情构造技术指标特征，并用“下一交易日是否上涨”作为分类标签；分类模型（随机森林 / XGBoost / Transformer）输出上涨概率，映射为 Buy/Hold/Sell。回归模型则预测未来 N 日收益（如 `ret_5d`），用于 Top-K 组合构建与分桶滚动回测。若开启联合 Transformer，会把指数成分股的数据合并训练一次，再对单股取最后窗口做预测。
