
import argparse
import json
import re
import torch
from spellchecker import SpellChecker


def main():
    
    #　集合型データで単語セットの入れ物を作成
    word_set = set()
    
    # COCOの学習用画像キャプションデータセットが格納されているファイルパス
    ##file_path = './Documents/MachineLearning/datasets/COCO/annotations/captions_train2017.json'
    
    # 引数で指定したファイルを読み込む
    ##with open(file_path, "r") as captions_file:
    with open('./self_made_captions/traffic_official_train_caption.txt', "r") as captions_file:
    
        # 読み込んだjsonファイル内の文字列をPython文字列に変換してdataに格納
        data = captions_file.readlines()
        
        
        # 単語辞書を作る
        for i in range(len(data)):
            
            # 1文の画像キャプションを選択して、単語に分割してリストに変換
            words = re.findall(r"[\w']+|[.,!?;]", data[i])
            
            
            # 集合型のword_setに和集合の演算を行う
            for word in words:
                
                # 1単語のみで構成される集合と、単語の集合で和集合の演算をする
                word_set.add(word)
                
    
    # データセットの誤字訂正のために使用するインスタンスを作成
    spell = SpellChecker(distance=1)
                
    print('誤字の修正をします')
    correct_word_set = set([spell.correction(word).lower() for word in word_set])
    
    print('データセットに含まれていた語彙数：', len(word_set))
    print('誤字修正後のデータセットの語彙数：', len(correct_word_set))
    
    # 誤字修正を行った語彙の集合をリスト型に変換
    word_list = list(correct_word_set)

    # 0から語彙数までの番号を単語IDとしてリストに格納
    ids = [i for i in range(len(word_list))]
    
    
    # 単語がkey、単語IDが値の辞書を作成
    stoi = dict(zip(word_list, ids))
    print('{単語:単語ID}の辞書型stoiを作成しました')
    print(stoi)
    
    # 単語が値、単語IDがkeyの辞書を作成
    itos = dict(zip(ids, word_list))
    print('{単語ID:単語}の辞書型itosを作成しました')
    print(itos)
    
    
    # 文頭と文末を意味する単語とその単語IDを定義
    stoi['<s>'] = len(stoi)
    stoi['</s>'] = len(stoi)
    itos[len(itos)] = '<s>'
    itos[len(itos)] = '</s>'

    
    # 単語IDと単語の正引き/逆引き辞書をさらに辞書型の要素として定義
    dicts = {'stoi': stoi,
             'itos': itos}
    
    # 第二引数に指定したパスに第一引数の変数を保存
    torch.save(dicts, './weights/dicts')
    print('単語辞書の保存が完了しました')

    
    
# メインモジュールとしてこのファイルを実行する
if __name__ == "__main__":
    
    main()
    