import os
import re
import shutil
# import tomllib

import numpy as np
import yaml


# yamlの設定ファイルの読み込み
def load_config():

    with open("../config/config.yaml", "rb") as f:
        config = yaml.safe_load(f)
        # print(config)
        # print("launch json modified")
    # print("mapping_namelist", config["class_mapping"]["name_list"])

    print("config file loaded")

    return config

# クラスの再定義を行った後の格納フォルダを作成
def make_processed_dir(config):

    dir_path_processed = config["path_processed_data"]
    list_class_mapping = config["class_mapping"]["name_list"]

    for class_name in list_class_mapping:

        dir_path = os.path.join(dir_path_processed, class_name)
        # print(dir_path)

        try:
            os.makedirs(dir_path, exist_ok = False)
        except:
            print(f"{dir_path} is already exits.")
        # print(dir_path)

    print("processed dir made")


def wirdata_mapping_and_copy(config):

    dict_wir_class_mapping = config["WIR_class_mapping"]
    dir_path_WIR = config["path_raw_data"]["dir_path_WIR"]
    dir_path_processed_data = config["path_processed_data"]

    # リーク防止。共通テストのファイルを除外するために、共通テストのファイル名一覧を作成
    dir_path_split_extracted_data = config["path_split_extracted_data"]["common_test"]
    class_name = config["class_mapping"]["name_list"]

    list_file_names_commontest = []

    for cn in class_name:
        dir_target = os.path.join(dir_path_split_extracted_data, cn)

        list_file_names_commontest.extend([
            f for f in os.listdir(dir_target)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])
        # print(dir_target)

    # ファイルの数のカウント
    count = 0
    
    # print("check here!!!", dict_wir_class_mapping)

    for k in dict_wir_class_mapping:
        # print("check here!!! k =", k)

        if dict_wir_class_mapping[k] != "not use":
            target_dir = os.path.join(dir_path_WIR, k)

            # print("check here!!!target_dir = ", target_dir)

            list_copy_file_names = os.listdir(target_dir)

            # print("check here!!!list_copy_file_names = ", list_copy_file_names)

            for copy_file_name in list_copy_file_names:

                copy_from = os.path.join(target_dir, copy_file_name)

                # print("check here!!!copy_from: ", copy_from)

                # コピー元のファイル名がナンバリングだけの場合、クラスを統合した後にコンフリクトする可能性がある。
                # そのため元データセットの名称および、元クラスの名称をコピー後のファイル名につける
                
                after_copy_file_name = "wir" + k + copy_file_name

                copy_to = os.path.join(dir_path_processed_data, dict_wir_class_mapping[k], after_copy_file_name)

                # print("check here!!!copy_to: ", copy_to)
                
                # 共通テストのファイルに含まれるかどうかの判定
                if after_copy_file_name in list_file_names_commontest:
                    print(f"{after_copy_file_name} in common test." )
                
                else:
                    # 既存ファイルがあれば上書きしない
                    if not os.path.exists(copy_to):
                        # print("cp done")
                        shutil.copy(copy_from, copy_to)
                    else:
                        print(f"{copy_to} already exists")

                    # count += 1
    # print(copy_file_name, list_file_names_commontest)
    # print("copy_file_name!!!!!!!!!", "wir" + k + copy_file_name)

    print(f"Processed {count} images")


def wdficdata_mapping_and_copy(config):
    dict_wdfic_class_mapping = config["WDFIC_class_mapping"]
    dir_path_wdfic = config["path_raw_data"]["dir_path_WDFIC"]
    dir_path_processed_data = config["path_processed_data"]

    list_file_names = os.listdir(dir_path_wdfic)

    # TODO
    # リーク防止の共通テストのファイルリスト。この関数のそとで、別に関数を作って、その結果を、copyする関数に渡す？
    # リーク防止。共通テストのファイルを除外するために、共通テストのファイル名一覧を作成
    dir_path_split_extracted_data = config["path_split_extracted_data"]["common_test"]
    class_name = config["class_mapping"]["name_list"]

    list_file_names_commontest = []


    for cn in class_name:
        dir_target = os.path.join(dir_path_split_extracted_data, cn)

        list_file_names_commontest.extend([
            f for f in os.listdir(dir_target)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])
        # print(dir_target)

    # ファイル数のカウント
    count = 0
    # print("check here!!!list_file_names: ", list_file_names)

    # ファイル名からクラス名を取り出す
    for file_name in list_file_names:
        # matched = re.match(r"([a-zA-Z]+)(\d+)", file_name)
        raw_class_name = re.match(r"([a-zA-Z]+)(\d+)", file_name).group(1)

        copy_from = os.path.join(dir_path_wdfic, file_name)
        
        after_copy_file_name = "wdfic" + file_name

        # コピー先での、ファイル名のコンフリクト防止のため、データセット名(wdfic)をつける。
        copy_to = os.path.join(dir_path_processed_data, dict_wdfic_class_mapping[raw_class_name], after_copy_file_name)

         # if raw_class_name in list(dict_wdfic_class_extractinig.keys()):
            # コピー先での、ファイル名のコンフリクト防止のため、データセット名(wdfic)をつける。
            # copy_to = os.path.join(dir_path_extracted_data, dict_wdfic_class_extractinig[raw_class_name], "wdfic" + file_name)

        # 共通テストのファイルに含まれるかどうかの判定
        if after_copy_file_name in list_file_names_commontest:
            print(f"{after_copy_file_name} in common test." )
        
        else:
            # print(f"{after_copy_file_name} is not for test")
        # 既存ファイルがあれば上書きしない
            if not os.path.exists(copy_to):
                shutil.copy(copy_from, copy_to)
                # print("cp done")
            else:
                print(f"{copy_to} already exists")
        
        count += 1

    print(f"Processed {count} images")
    # print("check here!!!copy_to: ", copy_to)

def remove_invalid_images():
    # remove_images_list.csvとかのファイルパスを読み込んで、それを元に削除する。

    print("not yet")


def main():
    config = load_config()
    make_processed_dir(config)
    wirdata_mapping_and_copy(config)
    wdficdata_mapping_and_copy(config)

if __name__ == "__main__":
    main()