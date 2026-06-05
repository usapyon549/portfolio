import glob
import os
import time

import pandas as pd
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types


def load_config(path_file: str) -> dict:

    with open(path_file, "r") as f:
        config = yaml.safe_load(f)

    return config


# 元データのコメントをすべて読み込む
def load_raw_data(config: dict) -> pd.DataFrame:
    
    df = pd.read_csv(config["data"]["raw"]["file_path"])
    print("raw data loaded")

    return df


# gemini apiの1日あたりの利用制限があり、前回までの保存データを読み込む
def load_saved_data(config: dict) -> pd.DataFrame:

    # ファイルの一覧を取得
    list_file_paths = glob.glob(os.path.join(config["output"]["dir_labeled"], "*.csv"))

    df_saved  = pd.DataFrame()

    for file_path in list_file_paths:
        df = pd.read_csv(file_path)
        df_saved  = pd.concat([df_saved , df])
        df_saved .reset_index(drop = True, inplace = True)

    print("saved data loaded")

    return df_saved


def extract_target_data(df_raw_data: pd.DataFrame, df_saved_data: pd.DataFrame) -> pd.DataFrame:
    
    # 未処理のidを抽出
    list_notyet_ids = list(set(df_raw_data["id"].to_list()) - set(df_saved_data["id"].to_list()))

    # 未処理のコメントを取り出す。
    df_filterd = df_raw_data[df_raw_data["id"].isin(list_notyet_ids)]

    print("target ids extracted")

    return df_filterd


def make_prompt(comments: str) -> str:
    
    comments_csv = comments
    
    prompt = f"""
        あなたはSNSコメント分析の専門家です。

        以下のコメントをそれぞれ独立に読み、行政・税金に関する不満の種類を1つだけ分類してください。

        【重要ルール】
        - 各コメントは完全に独立して判断すること
        - 他のコメントの影響を受けないこと
        - 必ず指定フォーマットのみを出力すること
        - 説明・補足・前置きは禁止
        - 出力はCSV形式のみ

        【分類カテゴリ】
        1. 無駄遣い：税金の使い道・支出内容が無駄だと批判
        2. 不透明性：説明不足・意思決定の不明確さへの不満
        3. 不公平：負担や恩恵の偏りへの不満
        4. 強い批判：怒り・攻撃的表現が主軸（感情が中心の場合のみ）
        5. その他：中立・事実・軽い意見

        【出力ルール（最重要）】
        以下の形式のみを出力すること：

        <id>,<label>,<confidence>

        【出力例】
        1,無駄遣い,0.92
        2,不透明性,0.88
        3,不公平,0.95

        【厳格ルール】
        - ヘッダー禁止
        - JSON禁止
        - Markdown禁止
        - 余計な文章禁止
        - 空行禁止
        - 各行は必ず3要素
        - confidenceは0〜1の小数（厳密でなくてよい）

        【入力データ】
        {comments_csv}
        """

    return prompt
    

def retry(client: genai.Client, 
          model: str, 
          config: dict, 
          content: str) -> types.GenerateContentResponse:

    response = client.models.generate_content(
                    model = model, 
                    config = config,
                    contents = content
                )
    
    return response


def save_response(df_labeled: pd.DataFrame, file_path: str) -> None:

    df_labeled.to_csv(file_path, index=False)
    print(f"labeled data saved to {file_path}")


def call_api(config: dict, df_target_data: pd.DataFrame) -> None:

    load_dotenv()
    api_key = os.getenv("API_KEY")

    model = config["api_settings"]["model_name"]
    dict_generation_config = config["api_settings"]["generation_config"]

    client = genai.Client(api_key = api_key)

    # バッチサイズをもとに、バッチの先頭となるindexを取得
    batch_size = config["api_settings"]["batch_size"]
    list_batch_indexes = [i for i in range(0, len(df_target_data), batch_size)]

    # バッチごとにapiを叩く
    for i in list_batch_indexes:

        df = df_target_data[i:i+batch_size]
        comments_csv = df[["id","comment_text"]].to_csv(index=False)

        prompt = make_prompt(comments_csv)

        print("calling api start")
        response = client.models.generate_content(
                model = model, 
                config = dict_generation_config,
                contents = prompt
            )

        # lines = response.text.split("\n")

        lines = [
            line.strip()
            for line in response.text.split("\n")
            if line.strip()
        ]

        #レスポンスの行がバッチサイズと一致しない時に再試行
        # 簡易的にretry
        if len(lines) != batch_size:
            response = retry(client, model, config, prompt)
            # lines = response.text.split("\n")

            lines = [
                line.strip()
                for line in response.text.split("\n")
                if line.strip()
            ]

        #responseを”,”で分解してリストに入れる
        list_response_row = []
        for row in lines: 
            list_response_row.append(row.split(","))

        # 途中でエラーが起きた場合に備え、バッチごとに結果をcsvで保存
        # バッチごとにスタートのid起点でファイル名を分ける形で保存。
        # 変な挙動をしたバッチがあればあとでそこだけを削除できる
        file_path_to_save = f"../data/gemini_response/start_id_{df["id"].iloc[0]}.csv"

        df_labeled = pd.DataFrame(list_response_row, columns=["id","label","confidence"])
        
        save_response(df_labeled=df_labeled, file_path=file_path_to_save)

        # 動作確認で一回だけバッチを回す
        # break
        
        # apiの１分回の利用制限にひっかからないようにsleep
        print("1 min sleep starts to avoid api regulation")
        time.sleep(60)


def main():
    config = load_config(path_file="../config/config.yaml")
    df_raw_data = load_raw_data(config=config)
    df_saved_data =load_saved_data(config=config)
    df_target_data = extract_target_data(df_raw_data=df_raw_data, 
                                         df_saved_data=df_saved_data)
    call_api(config=config, df_target_data=df_target_data)


if __name__ == "__main__":
    main()