
import random
import argparse
import re
import os.path as path
import sys
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from nltk.translate.bleu_score import sentence_bleu
import model2 as model
from PIL import Image
import data_utils



## 固定変数の宣言

# ミニバッチサイズを定義
batch_size = 32

# 学習データセット数（画像と文章のセット数）を定義
training_data_number = 960

# コーパスの語彙数を定義
vocabulary = 1083

# 学習率を定義
learning_rate = 1e-7

# 学習回数（エポック数）を定義
epochs = 201

# causal 畳み込みネットワークの層数を定義
decoder_layers_number = 6

# 単語ベクトルの要素数を定義
word_embedding_size = 300

# バリデーションのデータセット数（画像と文章のセット数）を定義
validation_data_number = 10

#計算環境の指定
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')



# この関数をメイン関数として実行
def main():

    # 逆引き、正引きの単語辞書を作成
    stoi, itos = data_utils.get_dicts()

    # 単語辞書の大きさ、単語のベクトルの大きさなどを指定してインスタンスを生成
    cnn_cnn = model.CNN_CNN_CE(len(stoi), word_embedding_size, n_layers=decoder_layers_number, train_cnn=True)
    
    # インスタンス変数のデータをCPUメモリからGPUメモリ上に移動
    cnn_cnn = cnn_cnn.to(device)

    # エポックを指定して学習済みモデルのパラメータをロード
    ##cnn_cnn.load(90)

    # ネットワークの重みを更新するアルゴリズムを定義
    optimizer = optim.Adam([
        {'params': cnn_cnn.language_module.parameters()},
        {'params': cnn_cnn.prediction_module.parameters()},
        {'params': cnn_cnn.attention_module.parameters()},
    ], lr=1e-4, weight_decay=0.1e-5)

    
    # 損失関数：予測する要素（単語）としてのクロスエントロピー
    criterion = torch.nn.NLLLoss()

    
    ## -------------- ニューラルネットワークの学習----------------------
    
    # BLEU-SCOREを格納するリストを宣言
    bleu_score = []
    
    
    # 指定したエポック数だけ繰り返し処理を実行
    for e in range(epochs):
        
        # ジェネレータ関数からイテレータを作成
        trainloader = data_utils.training_dataloader(training_data_number, batch_size, stoi)
        
        
        # バッチサイズ分の繰り返し処理（イテレータからデータを取り出していく）
        for batch, (images, captions) in enumerate(trainloader):

            # 画像キャプションの学習データ/教師データを作成
            train_indices_v, expected_indices_v, train_labels, max_length= data_utils.make_train_caption_datasets(stoi, vocabulary, captions)
            
            # 学習用の画像と画像キャプションデータをCPUメモリ上からGPUメモリ上に移動
            images_v = images.to(device)
            train_indices_v = train_indices_v.to(device)
            expected_indices_v = expected_indices_v.to(device)
            
            # 正解データの画像キャプションの単語IDのインデックスを格納するためのリスト
            valid_training_indices = []
            
            
            # バッチサイズ分の学習用画像キャプションの単語に、順番に1から総単語数の番号を振る
            for i, label in enumerate(train_labels):
                
                # 各キャプションの単語数分リスト内で繰り返し処理で、インデックスのみ取り出す
                valid_training_indices = valid_training_indices + [j for j in range(i*(max_length), i*(max_length) + len(label))]
                ##print('valid_training_indices', valid_training_indices)

            
            # view(-1)で行列からベクトルに変換し、上で作成したインデックスで指定した位置の要素を取り出す
            valid_expected_v = expected_indices_v.view(-1)[valid_training_indices]
            ##print('\n expected_indices_v',expected_indices_v)
            ##print('\n valid_expected_v',valid_expected_v)

            # パラメータの勾配を全て初期化する
            optimizer.zero_grad()

            # 画像をニューラルネットワークに入力して単語の発生確率を出力
            outputs_v = cnn_cnn(images_v, train_indices_v)

            # 生成された画像キャプションの形状を変更
            outputs_v = outputs_v.view(-1, cnn_cnn.vocab_size)

            # 予測/正解キャプションを損失関数の引数にして、誤差を算出
            loss = criterion(outputs_v[valid_training_indices], valid_expected_v)

            # ニューラルネットワークの誤差逆伝播を実行
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(cnn_cnn.parameters(), 1.0)

            # 誤差逆伝播勾配をもとにニューラルネットワークのパラメータを更新
            optimizer.step()
            
            
        # BLUEスコアをバリデーションの度に初期化
        bleu_score_epoch = 0

        # バリデーション用の画像/画像キャプションをロード
        images_for_validation, captions_for_validation = data_utils.validation_dataloader(stoi)


        # バリデーションデータ数回繰り返し処理
        for i in range(validation_data_number):

            # 検証用の画像を選択
            img = images_for_validation[i]
            
            # 変数データをCPUメモリからGPUメモリに移動
            img = img.to(device)
            
            # 画像のサイズを（3, 224, 224）から（1, 3, 224, 224）に変換
            img = img.view(1, *img.shape)
            
            # 検証用の画像でキャプションを予測
            sentence = cnn_cnn.sample(img, stoi, itos)[0]

            # 正解画像キャプションを選択
            captions = captions_for_validation[i]

            print('\n 予測されたキャプション',sentence)
            print('正解のキャプション',captions)

            # sentence_bleuを計算
            bleu_score_epoch += sentence_bleu(captions, sentence, weights=(0.25, 0.25, 0.25, 0.25))

            
        # 検証用データでのBLEUスコアの平均値を算出
        bleu_score.append(bleu_score_epoch / validation_data_number)

        
        # blue_scoreに最大値のものが格納されたら、その時のモデルを保存する
        if(e%10==0 or (bleu_score[-1] == max(bleu_score))):

            # モデルを保存
            cnn_cnn.save(e)
        
                
            
# メインモジュールとしてこのファイルを実行
if __name__ == "__main__":
    
    # このファイルの上記に定義したmain関数を実行
    main()
