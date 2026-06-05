# 標準
import os
import pickle

# サードパーティ
import pandas as pd
import yaml
import lightgbm as lgb


def load_config(path_config: str) -> dict:
    # docstring
    # config読む
    with open(path_config, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_data(path_file: str) -> pd.DataFrame:
    return pd.read_csv(path_file)


def train_model(config: dict) -> lgb.LGBMRegressor:

    df_train = load_data(config["data"]["processed"]["path_train"])
    df_val = load_data(config["data"]["processed"]["path_val"])

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    target_col = config["features"]["target_col"]
    feature_cols = config["features"]["feature_cols"]

    X_train = df_train[feature_cols]
    y_train =df_train[target_col]

    X_val = df_val[feature_cols]
    y_val = df_val[target_col]

    model_lgb = lgb.LGBMRegressor(
        **config["model"]["params"]
    )

    eval_set = [(X_val, y_val)]

    print("train starts")

    model_lgb.fit(
        X_train, 
        y_train,
        eval_set = eval_set,
        # verbose=-1 # 学習ログを省略
    )

    print("train done")

    return model_lgb


def save_model(config: dict, model_lgb: lgb.LGBMRegressor) -> None:

    # モデルの保存
    # ファイルの保存先を作成
    os.makedirs(os.path.dirname(config["output"]["path_model"]), exist_ok=True)

    with open(config["output"]["path_model"], "wb") as f:
        pickle.dump(model_lgb, f)

    print(f'model saved to {config["output"]["path_model"]}')


def main():
    config = load_config(path_config = "../config/config.yaml")
    model_lgb = train_model(config = config)
    save_model(config=config, model_lgb=model_lgb)

if __name__ == "__main__":
    main()