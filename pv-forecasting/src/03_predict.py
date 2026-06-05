# 標準
import os
import pickle

# サードパーティ
import numpy as np
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


def load_model(path_file: str) -> lgb.LGBMRegressor:
    with open(path_file, "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model
   

def predict(df_features: pd.DataFrame, config: dict, model: lgb.LGBMRegressor) -> np.ndarray:
    
    df_features = df_features.reset_index(drop=True)
    feature_cols = config["features"]["feature_cols"]

    return model.predict(df_features[feature_cols])


def save_prediction(config: dict, y_pred: np.ndarray) -> None:

    # ファイルの保存先を作成
    os.makedirs(os.path.dirname(config["output"]["path_pred"]), exist_ok=True)

    df_pred = pd.DataFrame({"predicted pv": y_pred})
    df_pred.to_csv(config["output"]["path_pred"], index=False)

    print(f'prediction saved to {config["output"]["path_pred"]}')

    
def main():
    config = load_config(path_config = "../config/config.yaml")
    df = load_data(path_file = config["data"]["processed"]["path_test"])
    model = load_model(path_file = config["output"]["path_model"])
    predictions = predict(df_features=df, config=config, model=model)
    save_prediction(config=config,y_pred=predictions)


if __name__ == "__main__":
    main()