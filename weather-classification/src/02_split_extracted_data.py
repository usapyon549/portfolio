import os
import random
import shutil

import yaml


def load_config(path_file: str) -> dict:

    with open(path_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def make_dir(dir_path: str, class_name: str) -> None:

    dir_path = os.path.join(dir_path, class_name)

    # 共通テストデータを誤って上書きしないよう、
    # 既存ディレクトリが存在する場合は新規作成しないを明示
    try:
        os.makedirs(dir_path, exist_ok = False)
    except FileExistsError:
        print(f"{dir_path} already exists. Skip creating directory.")


# 分割したデータを格納するフォルダを作成
def make_split_dir(config: dict) -> None:

    dir_path_train = config["path_split_extracted_data"]["train"]
    dir_path_val = config["path_split_extracted_data"]["val"]
    dir_path_common_test =  config["path_split_extracted_data"]["common_test"]

    list_class_mapping = config["class_mapping"]["name_list"]

    for class_name in list_class_mapping:

        make_dir(dir_path=dir_path_train, class_name=class_name)
        make_dir(dir_path=dir_path_val, class_name=class_name)
        make_dir(dir_path=dir_path_common_test, class_name=class_name)


    print("split dir prepared")


# ファイルコピー用の関数
def file_copy(list_file_names: list, dir_path_from: str, dir_path_to: str) -> tuple[int, int]:

    copied = 0
    skipped = 0

    for f in list_file_names:

        copy_from = os.path.join(dir_path_from, f)
        copy_to = os.path.join(dir_path_to, f )

        # 既存ファイルがあれば上書きしない
        if not os.path.exists(copy_to):
            shutil.copy(copy_from, copy_to)
            copied += 1
            # print("cp done")
        else:
            skipped += 1
            print(f"{copy_to} already exists")
    

    # print("FROM:", copy_from, "TO:", copy_to)

    return copied, skipped


def split_and_copy(config: dict) -> None:
    
    random.seed(config["random_seed"])

    # 分割する元データの親ディレクトリ
    dir_path_extracted_data = config["path_extracted_data"]

    # 分割したあとにデータを格納する親ディレクトリ
    dir_path_train = config["path_split_extracted_data"]["train"]
    dir_path_val = config["path_split_extracted_data"]["val"]
    dir_path_common_test =  config["path_split_extracted_data"]["common_test"]

    list_class_mapping = config["class_mapping"]["name_list"]
    
    copied_total = 0
    skipped_total = 0

    # クラスごとにデータの分割
    for class_name in list_class_mapping:
        
        dir_path_raw_data = os.path.join(dir_path_extracted_data, class_name) 
        
        # list_file_names = os.listdir(dir_path_raw_data)

        list_file_names = [
            f for f in os.listdir(dir_path_raw_data)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        # リストをランダムにシャッフル
        random.shuffle(list_file_names)
        
        # train:val:common_test = 8:1:1の割合で作る
        file_num = len(list_file_names)
        slicer1 = file_num * 80//100
        slicer2 = file_num * 90//100
        
        list_file_names_train = list_file_names[:slicer1]
        list_file_names_val = list_file_names[slicer1:slicer2]
        list_file_names_common_test = list_file_names[slicer2:]

        # 分割が正常に機能しているか、
        # ファイルの数が合っているかの確認
        assert file_num == (
            len(list_file_names_train)
            + len(list_file_names_val)
            + len(list_file_names_common_test)
        )

        # trainデータの作成
        copied, skipped = file_copy(list_file_names=list_file_names_train, 
                                    dir_path_from=dir_path_raw_data, 
                                    dir_path_to=os.path.join(dir_path_train, class_name))
        copied_total += copied
        skipped_total += skipped

        # 検証データの作成
        copied, skipped = file_copy(list_file_names=list_file_names_val, 
                                    dir_path_from=dir_path_raw_data, 
                                    dir_path_to=os.path.join(dir_path_val, class_name))
        copied_total += copied
        skipped_total += skipped

        # テストデータの作成
        copied, skipped = file_copy(list_file_names=list_file_names_common_test, 
                                    dir_path_from=dir_path_raw_data, 
                                    dir_path_to=os.path.join(dir_path_common_test, class_name))
        copied_total += copied
        skipped_total += skipped
        
        # コピーしたファイルの数の確認→デバッグの時につかったから、あとでけす。
        # print("class name:", class_name)
        # print("raw:", file_num)
        # print("train:", len(list_file_name_train))
        # print("val:", len(list_file_name_val))
        # print("test:", len(list_file_name_common_test))
        # print("sum:", len(list_file_name_train) + len(list_file_name_val) + len(list_file_name_common_test))
        # print("actual files:", len(os.listdir(dir_path_raw_data)))

    print(f"copied:{copied_total}, skipped:{skipped_total}")   


def main():
    config = load_config(path_file="../config/config.yaml")

    # 共通テストセット固定のため、
    # 通常は再実行しない
    make_split_dir(config)
    split_and_copy(config) 


if __name__ == "__main__":
    main()