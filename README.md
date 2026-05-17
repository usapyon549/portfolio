# Portfolio

数値・画像・テキストといった異なるデータを用いて、  
予測・分類・変化検知を行った機械学習ポートフォリオです。

単純に精度を追うだけでなく、

- なぜその結果になったのか
- どの特徴量が支配的なのか
- どの改善が有効 / 無効だったのか

を分析・解釈することを重視しました。

また、実務を意識し、

- 限られた計算資源
- データ数の制約
- モデルの再現性

なども考慮して設計しています。

---

# Projects

## 1. pv-forecasting（メイン）

気象データを用いた太陽光発電量の時系列予測。

LightGBM + Optuna を用いてモデル改善を行い、  
SHAPによるモデル解釈や残差分析を通じて、

> 「モデルが何に依存して予測しているか」

を分析しました。

### 主なポイント

- 線形回帰 → LightGBM による精度改善
- sin/cos特徴量による周期性検証
- SHAPによる特徴量寄与分析
- GHI（全天日射量）依存構造の分析
- Optunaによるハイパーパラメータ最適化

📁 [`pv-forecasting`](./pv-forecasting)

---

## 2. weather-classification

天候画像（晴れ・曇り・雨・雪）を分類する画像分類プロジェクト。

太陽光発電における外生要因として利用することを想定し、  
EfficientNet-B0 を用いて分類モデルを構築しました。

また、

- データ拡張
- optimizer変更
- class weight

などを比較し、

> 「モデル改善が常に性能向上につながるわけではない」

ことを検証しました。

### 主なポイント

- EfficientNet-B0による転移学習
- Adam / SGD 比較
- augmentation効果検証
- class weight効果検証
- 軽量環境（MacBook Air）での実験設計

📁 [`weather-classification`](./weather-classification)

---

## 3. sns-comment-analysis

ニュースサイトのコメントを対象に、  
LLMを用いたコメント分類と時系列分析を行ったプロジェクト。

行政・税金系ニュースに対する反応を分析し、

- 感情的批判へ収束するケース
- 議論が分散するケース
- 一時的反応で終わるケース

など、記事ごとの反応構造の違いを観察しました。

**＊特定の政治思想や立場を分析することを目的としたものではなく、  
あくまで、コメント数が比較的多く、幅広いユーザー反応を観察しやすい行政・公共政策系ニュースを対象として
選定しています。**

### 主なポイント

- Gemini API によるコメント分類
- CSV形式によるバッチ推論
- ラベル分布分析
- コメント傾向の時系列変化分析
- 人手ラベルとの比較

📁 [`sns-comment-analysis`](./sns-comment-analysis)

---

# 技術スタック

| 分類 | 技術 |
|---|---|
| 言語 | Python |
| データ処理 | pandas / numpy |
| 可視化 | matplotlib |
| 時系列予測 | LinearRegression / LightGBM / Optuna |
| 画像分類 | TensorFlow / EfficientNet |
| LLM | Gemini API |
| 解釈性 | SHAP |

---

# ディレクトリ構成

```plaintext
portfolio/
├── pv-forecasting/
├── weather-classification/
├── sns-comment-analysis/
└── README.md
```

---

# Additional Experiments

portfolio とは別に、  
技術検証用リポジトリ `ml-lab` も運用しています。

現在は、

- MLflowを活用した、軽量MLOps
- Docker 化
- SageMaker deploy
- 軽量 RAG

など、技術の検証を進めています。