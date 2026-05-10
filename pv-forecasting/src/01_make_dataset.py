import os
from datetime import datetime

import numpy as np
import pandas as pd
import yaml


def load_config(path_config: str) -> dict:
    # docstring
    # config読む
    with open(path_config, "r") as f:
        config = yaml.safe_load(f)

    # print("config file loaded")

    return config


def load_data(config: dict) -> pd.DataFrame:
    # docstring
    # configを受け取って、元データを読み込む。dfを返す
    
    path_filedir = config["data"]["raw"]["path_weather"]
    read_cols = config["data"]["raw"]["read_cols"]
    df_weather = pd.DataFrame()

    list_files = os.listdir(path_filedir)
    
    # print(list_files)
    dfs = []    
    for file_name in list_files:
        file_path = os.path.join(path_filedir, file_name)
        df = pd.read_csv(file_path, skiprows=2, usecols=read_cols)
        dfs.append(df)

    df_weather = pd.concat(dfs, ignore_index=True)
    df_weather = df_weather.reset_index(drop = True)

    # print("check here:", df_weather.head(2))

    return df_weather


def make_features(config: dict, df:pd.DataFrame) -> pd.DataFrame:
    
    df_features = df.copy()

#   datetimeを作成。元データのタイムゾーンはutc
    df_features["utc"] = pd.to_datetime(df_features[["Year", "Month", "Day", "Hour"]])

    # jstに変換
    df_features["jst"] = df_features["utc"] + pd.Timedelta(hours=9)

#   day_of_yearを取得
    df_features["day_of_year"] = df_features["jst"].dt.dayofyear
    is_leap = df_features["jst"].dt.is_leap_year
    max_days = np.where(is_leap, 366, 365)
    normalized_day_of_year = (df_features["day_of_year"] - 1) / max_days
    # day_of_yearのsin/cos変換
    df_features["day_of_year_sin"] = np.sin(2 * np.pi * normalized_day_of_year.reset_index(drop=True))
    df_features["day_of_year_cos"] = np.cos(2 * np.pi * normalized_day_of_year.reset_index(drop=True))

    df_features["hour_jst"] = df_features["jst"].dt.hour
    # 冗長かもしれないけど、うるう年の操作があるday_of_yearとは分けて、sincos化
    df_features["hour_jst_sin"] = np.sin(2 * np.pi * df_features["hour_jst"] / 24)
    df_features["hour_jst_cos"]  = np.cos(2 * np.pi * df_features["hour_jst"] / 24)

    # 発電量[Wh/m2] = GHI ＊ 1 ＊ 0.18 * (1 - 0.004 * (Temperature - 25)) * 1
    df_features["pv"] = df_features["GHI"] * 1 * 0.18 * (1 - 0.004 * (df_features["Temperature"] - 25))  * 1

    return df_features


def make_dataset(config: dict, 
                 df:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
   
    df_final = df.copy()

    use_cols = [config["features"]["target_col"]] + config["features"]["feature_cols"]

    # train, val, testを時系列で分割
    train_start = datetime(2016, 1, 1, 0, 0, 0)
    val_start = datetime(2019, 1, 1, 0, 0, 0)
    test_start = datetime(2020, 1, 1, 0, 0, 0)

    df_train = df_final[use_cols][(df_final["jst"] >= train_start) 
                                            & (df_final["jst"] < val_start)]
    df_val = df_final[use_cols][(df_final["jst"] >= val_start) 
                                            & (df_final["jst"] < test_start)]
    df_test =  df_final[use_cols][df_final["jst"] >= test_start]

    return df_train, df_val, df_test


def save_data(config: dict, 
              df_train: pd.DataFrame, 
              df_val: pd.DataFrame, 
              df_test: pd.DataFrame) -> None:


    # ファイルの保存先を作成
    os.makedirs(os.path.dirname(config["data"]["processed"]["path_train"]), exist_ok=True)
    
    # 分割したデータを保存
    df_train.to_csv(config["data"]["processed"]["path_train"], index=False, encoding = "utf-8")
    df_val.to_csv(config["data"]["processed"]["path_val"], index=False, encoding = "utf-8")
    df_test.to_csv(config["data"]["processed"]["path_test"], index=False, encoding = "utf-8")

    print("data files saved")


def main():
    config = load_config(path_config="../config/config.yaml")
    df = load_data(config=config)
    df = make_features(config=config, df=df)
    df_train, df_val, df_test = make_dataset(config=config, df=df)
    save_data(config=config, df_train=df_train, df_val=df_val, df_test=df_test)

if __name__ == "__main__":
    main()
