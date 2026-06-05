import os
import pickle

import tensorflow as tf
import tensorflow.keras.backend as K
import yaml

from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model


def load_config(path_file: str) -> dict:

    with open(path_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_data(config: dict) -> tuple[tf.data.Dataset, tf.data.Dataset]:

    seed = config["random_seed"]

    dir_path_train = config["path_split_extracted_data"]["train"]
    dir_path_val = config["path_split_extracted_data"]["val"]

    # 画像サイズは、efficientnet b0の想定
    img_height = config["load_file_settings"]["img_height"]
    img_width = config["load_file_settings"]["img_width"]

    # バッチサイズは小さめ
    batch_size = config["load_file_settings"]["batch_size"]

    # 学習データの読み込み
    dataset_train = tf.keras.utils.image_dataset_from_directory(
        dir_path_train,
        # subset = "training",    # 学習用    
        seed = seed, 
        image_size = (img_height, img_width),
        batch_size=batch_size,
        shuffle=True
    )

    # 検証データの読み込み
    dataset_val = tf.keras.utils.image_dataset_from_directory(
        dir_path_val,
        # subset = "val",    # 検証用    
        seed = seed, 
        image_size = (img_height, img_width),
        batch_size=batch_size,
        shuffle=False
    )

    return dataset_train, dataset_val


def train_model(config: dict, 
                dataset_train: tf.data.Dataset, 
                dataset_val:tf.data.Dataset) -> tuple[tf.keras.Model, dict]:
    
    seed = config["random_seed"]

    # 画像サイズは、efficientnet b0の想定
    img_height = config["load_file_settings"]["img_height"]
    img_width = config["load_file_settings"]["img_width"]

    epoch = config["train_settings"]["epoch"]
 
    # 乱数設定
    tf.keras.utils.set_random_seed(seed)

    # 計算結果の再現性を担保するため、TensorFlowの「決定論的動作」を設定。
    tf.config.experimental.enable_op_determinism()

    # ImageNetで学習済みのEfficientNetB0を読み込み（分類ヘッド除く）
    base_model = EfficientNetB0(weights="imagenet", 
                                include_top=False, 
                                input_shape=(img_height, img_width, 3))


    # トップ層のみ学習したいため、層を凍結
    base_model.trainable = False

    # モデルの構築
    inputs = layers.Input(shape=(img_height, img_width, 3))
    x = base_model(inputs, training=False) #base_model.trainable=Falseしているが、BatchNormalization層を固定するためにここでもfalse
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(len(dataset_train.class_names), 
                           activation="softmax")(x) #多分類なのでsoftmax

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("train starts")

    history = model.fit(
        x=dataset_train, 
        epochs=epoch, 
        # epochs=1, #動作確認用
        validation_data=dataset_val,
        )

    print("train ends")

    return model, history.history


def save_model(config: dict, model: tf.keras.Model) -> None:

    # ファイルの保存先を作成
    os.makedirs(os.path.dirname(config["output"]["path_model"]), exist_ok=True)

    model.save(config["output"]["path_model"])

    print(f'model saved to {config["output"]["path_model"]}')

def save_history(config: dict, history: dict) -> None:

    # ファイルの保存先を作成
    os.makedirs(os.path.dirname(config["output"]["path_history"]), exist_ok=True)

    with open(config["output"]["path_history"], 'wb') as f:
        pickle.dump(history, f)

    print(f'history saved to {config["output"]["path_history"]}')

def main():
    config = load_config(path_file="../config/config.yaml")
    dataset_train, dataset_val = load_data(config=config)
    model, history = train_model(config=config, dataset_train=dataset_train, dataset_val=dataset_val)
    save_model(config=config, model=model)
    save_history(config=config, history=history) 

    # tensorflowが確保したメモリをリリース
    K.clear_session()

if __name__ == "__main__":
    main()