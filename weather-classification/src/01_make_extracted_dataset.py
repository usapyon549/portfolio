import os
import re
import shutil

import yaml


# yamlの設定ファイルの読み込み
def load_config(path_file: str) -> dict:

    with open(path_file, "r") as f:
        config = yaml.safe_load(f)

    return config


# クラスごとに取り出した画像を格納するフォルダを作成
def make_extracted_dir(config: dict) -> None:

    dir_path_extracted = config["path_extracted_data"]
    list_class_mapping = config["class_mapping"]["name_list"]

    for class_name in list_class_mapping:

        dir_path = os.path.join(dir_path_extracted, class_name)

        # 共通テストデータを誤って上書きしないよう、
        # 既存ディレクトリが存在する場合は新規作成しない
        try:
            os.makedirs(dir_path, exist_ok = False)
        except FileExistsError:
            print(f"{dir_path} already exists. Skip creating directory.")
        # print(dir_path)

    print("extracted dir made")


def extract_wir_data(config: dict) -> None:

    dict_wir_class_extracting = config["WIR_class_extracting"]
    dir_path_WIR = config["path_raw_data"]["dir_path_WIR"]
    dir_path_extracted_data = config["path_extracted_data"]

    # ファイルの数のカウント
    count = 0
    
    for src_class, dst_class in dict_wir_class_extracting.items():

        if dst_class != "not use":

            target_dir = os.path.join(dir_path_WIR, src_class)
            list_copy_file_names = os.listdir(target_dir)

            for copy_file_name in list_copy_file_names:

                copy_from = os.path.join(target_dir, copy_file_name)

                # コピー元のファイル名がナンバリングだけの場合、クラスを統合した後にコンフリクトする可能性がある。
                # そのため元データセットの名称および、元クラスの名称をコピー後のファイル名につける
                copy_to = os.path.join(dir_path_extracted_data, dst_class, "wir" + src_class + copy_file_name)

                # break

                # 後のモデル比較を想定し、誤って元画像が入れ替わらないように
                # 既存ファイルが存在する場合は上書きしない
                if not os.path.exists(copy_to):
                    # print("cp done")
                    shutil.copy(copy_from, copy_to)
                    count += 1
                else:
                    print(f"{copy_to} already exists. Skip copy.")


    print(f"Processed {count} images")


def extract_wdfic_data(config: dict) -> None:
    dict_wdfic_class_extracting = config["WDFIC_class_extracting"]
    dir_path_wdfic = config["path_raw_data"]["dir_path_WDFIC"]
    dir_path_extracted_data = config["path_extracted_data"]

    list_file_names = os.listdir(dir_path_wdfic)

    # ファイル数のカウント
    count = 0

    # ファイル名からクラス名を取り出す
    for file_name in list_file_names:
        # matched = re.match(r"([a-zA-Z]+)(\d+)", file_name)
        # raw_class_name = re.match(r"([a-zA-Z]+)(\d+)", file_name).group(1)

        match = re.match(r"([a-zA-Z]+)(\d+)", file_name)
        if not match:
            continue

        raw_class_name = match.group(1)

        copy_from = os.path.join(dir_path_wdfic, file_name)
        
        if raw_class_name in dict_wdfic_class_extracting:

            # コピー先での、ファイル名のコンフリクト防止のため、データセット名(wdfic)をつける。
            copy_to = os.path.join(dir_path_extracted_data, dict_wdfic_class_extracting[raw_class_name], "wdfic" + file_name)

        # break
            # 後のモデル比較を想定し、誤って元画像が入れ替わらないように
            # 既存ファイルが存在する場合は上書きしない
            if not os.path.exists(copy_to):
                shutil.copy(copy_from, copy_to)
                count += 1

                # print("cp done")
            else:
                print(f"{copy_to} already exists. Skip copy.")

        # else:
        #     print(f"{raw_class_name} is not target")
        

    print(f"Processed {count} images")
    # print("check here!!!copy_to: ", copy_to)

# def remove_invalid_images():
    # remove_images_list.csvとかのファイルパスを読み込んで、それを元に削除する。

    # print("not yet")


def main():
    config = load_config(path_file="../config/config.yaml")
    make_extracted_dir(config=config)
    extract_wir_data(config=config)
    extract_wdfic_data(config=config)


if __name__ == "__main__":
    main()