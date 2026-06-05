import os

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml


def load_config(path_file: str) -> dict:

    with open(path_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_data(config: dict) -> tf.data.Dataset:

    seed = config["random_seed"]
    dir_path_test = config["path_split_extracted_data"]["common_test"]

    # 画像サイズは、efficientnet b0の想定
    img_height = config["load_file_settings"]["img_height"]
    img_width = config["load_file_settings"]["img_width"]

    # バッチサイズは小さめ
    batch_size = config["load_file_settings"]["batch_size"]

    # 共通テストデータの読み込み
    dataset_test = tf.keras.utils.image_dataset_from_directory(
        dir_path_test,
        # subset = "test",    # テスト用    
        seed = seed, 
        image_size = (img_height, img_width),
        batch_size=batch_size,
        shuffle=False
    )

    return dataset_test


def predict_classes(config: dict, dataset_test: tf.data.Dataset) -> np.ndarray:

    model = tf.keras.models.load_model(config["output"]["path_model"])

    y_probs = model.predict(dataset_test)
    
    # 最大確率のクラスを予測ラベルとする
    y_pred = np.argmax(y_probs, axis=-1)

    # print(type(y_pred))

    return y_pred


def save_result(config: dict, y_pred: np.ndarray) -> None:

    # ファイルの保存先を作成
    os.makedirs(os.path.dirname(config["output"]["path_result"]), exist_ok=True)

    df_pred = pd.DataFrame({"class": y_pred})
    df_pred.to_csv(config["output"]["path_result"], index=False)

    print(f'classification result saved to {config["output"]["path_result"]}')


def main():
    config = load_config(path_file="../config/config.yaml")
    dataset_test = load_data(config=config)
    y_pred = predict_classes(config=config, dataset_test=dataset_test)
    save_result(config=config, y_pred=y_pred)


if __name__ == "__main__":
    main()