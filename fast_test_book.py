
import matplotlib.pyplot as plt 
import numpy as np 
import torch
from PIL import Image
import torchvision.transforms as transforms
import re
from rouge import Rouge 
from nltk.translate.bleu_score import corpus_bleu
import data_utils
import model2 as model



## テストする学習済みモデルを調整するパラメータ

# テストを開始するエポック数
start_epoch = 0

# 何エポックまでのモデルを評価するか
test_to_what_epoch = 201

# 精度を図るモデルを何エポックごとに選択するか（すべてのモデルでテストする場合は1）
test_sample_interval = 10

# 何枚の画像でblue/rougeを計算するか（上限20枚）
number_of_test_images = 20


# 計算環境の指定
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# テスト結果を詳細な数値で保存するファイルパスを指定
detailed_value_file_path = './test_result_detailed_value.txt'

# テスト結果を折線グラフで保存するファイルパスを指定
result_graph_file_path  = './test_result_graph_figure.jpg'

# テスト用の画像と画像キャプションを作成
images_for_test, self_made_test_caption = data_utils.test_dataloader()

# 画像データをGPUメモリ上に移動
images_for_test = images_for_test.to('cuda')

# 単語辞書を作成
stoi, itos = data_utils.get_dicts()



def main():
    
    ## グラフの軸を定義

    # x軸のリスト
    horizontal_axis_list = np.array([])

    # y軸のリスト
    rouge1_f_score_list = np.array([])
    rouge2_f_score_list = np.array([])
    rougeL_f_score_list = np.array([])
    
    
    
    # 確認するepochの段階数を指定
    for epoch in range(start_epoch, test_to_what_epoch, test_sample_interval):

        # CNN_CNN_CEクラスでインスタンス作成
        cnn = model.CNN_CNN_CE(len(stoi), 300, n_layers=6, max_length=30).cuda()

        # 学習したエポック地点を選択してモデルパラメータを器に格納
        cnn.load(epoch)

        # テスト用にパラメータを固定
        cnn.eval()
        
        # 読み込んだ学習済みモデルの学習回数を表示
        print('\n[ 学習 {} 回目のモデルを読み込みました ]'.format(epoch))

        
        # rougeスコアを格納するリストを宣言
        rouge_scores = []


        # テスト画像20枚で一枚ずつテスト
        for i in range(number_of_test_images):

            # 画像を1枚選択
            img = images_for_test[i]
            
            # 画像ファイルのサイズを（3, 224, 224）→（1, 3, 224, 224）に変換
            img = img.view(1, *img.shape)

            # 正解の画像キャプション
            target = self_made_test_caption[i]

            # 予測された画像キャプション
            sentence = cnn.sample(img, stoi, itos)[0]

            # 予測した単語リストの要素を連結
            words = " ".join(sentence)

            # ROUGEスコアを算出するメソッドを持ったインスタンスを作成
            rouge = Rouge()
            
            # ROUGEスコアを計算
            rouge_score = rouge.get_scores(words, target)

            # キャプションが生成された画像の番号と予測/正解キャプションを表示
            print('\nテスト画像番号：',i)
            print('予測画像キャプション：',words)
            print('正解画像キャプション：',target)

            # ROUGEスコアを格納するリストに追加
            rouge_scores.append(rouge_score)


        # ROUGEスコアを初期化
        total_rouge1_f_score = 0
        total_rouge2_f_score = 0
        total_rougeL_f_score = 0


        # テストデータ数分で平均値を算出するためにROUGEスコアを合計
        for score in range(start_epoch, number_of_test_images):

            # ROUGE-1のf値の合計を計算
            total_rouge1_f_score = total_rouge1_f_score + rouge_scores[score][0]['rouge-1']['f']

            # ROUGE-2のf値の合計を計算
            total_rouge2_f_score = total_rouge2_f_score + rouge_scores[score][0]['rouge-2']['f']

            # ROUGE-Lのf値の合計を計算
            total_rougeL_f_score = total_rougeL_f_score + rouge_scores[score][0]['rouge-l']['f']


        # ROUGE-1のf値の平均を計算
        average_rouge1_f_score =total_rouge1_f_score/len(rouge_scores)

        # ROUGE-2のf値の平均を計算
        average_rouge2_f_score =total_rouge2_f_score/len(rouge_scores)

        # ROUGE-Lのf値の平均を計算
        average_rougeL_f_score =total_rougeL_f_score/len(rouge_scores)


        # y軸方向の値に対応するROUGE-1スコアリストに要素追加
        rouge1_f_score_list = np.append(rouge1_f_score_list, average_rouge1_f_score)

        # y軸方向の値に対応するROUGE-2スコアリストに要素追加
        rouge2_f_score_list = np.append(rouge2_f_score_list, average_rouge2_f_score)

        # y軸方向の値に対応するROUGE-Lスコアリストに要素追加
        rougeL_f_score_list = np.append(rougeL_f_score_list, average_rougeL_f_score)

        # x軸方向に対応するリストに要素追加
        horizontal_axis_list = np.append(horizontal_axis_list, int(epoch))

    
    ## 各エポックでの学習済みモデルの比較結果をまとめる

    # ROUGE-1/2/Lの最大値を取得
    max_rouge1_score = max(rouge1_f_score_list)
    max_rouge2_score = max(rouge2_f_score_list)
    max_rougeL_score = max(rougeL_f_score_list)
    
    # ROUGE-1/2/Lが最大値を取るエポックを取得
    max_rouge1_score_epoch = np.argmax(rouge1_f_score_list)
    max_rouge2_score_epoch = np.argmax(rouge2_f_score_list)
    max_rougeL_score_epoch = np.argmax(rougeL_f_score_list)
    
    
    # テスト結果のまとめを表示
    print('\n\n[ テスト結果 ]')
    
    print('\nrouge1の最大値：', max_rouge1_score)
    print('rouge1で最大値を取るepoch：', max_rouge1_score_epoch*test_sample_interval)
    
    print('\nrouge2の最大値：', max_rouge2_score)
    print('rouge2で最大値を取るepoch：', max_rouge2_score_epoch*test_sample_interval)
    
    print('\nrougeLの最大値：', max_rougeL_score)
    print('rougeLで最大値を取るepoch：', max_rougeL_score_epoch*test_sample_interval)
    

    # 指定したテキストファイルにrouge1/2/Lの最大スコアとそのときのエポックを書き込む
    with open(detailed_value_file_path, mode='a') as f:

        f.write('\n\n最大のrouge1スコア='+ '{}'.format(max_rouge1_score))
        f.write('\n最大のrouge2スコア='+'{}'.format(max_rouge2_score))
        f.write('\n最大のrougeLスコア='+'{}'.format(max_rougeL_score))

        f.write('\n\n最大のrouge1スコアのエポック='+'{}'.format(max_rouge1_score_epoch*test_sample_interval))
        f.write('\n最大のrouge2スコアのエポック='+'{}'.format(max_rouge2_score_epoch*test_sample_interval))
        f.write('\n最大のrougeLスコアのエポック='+'{}'.format(max_rougeL_score_epoch*test_sample_interval))



    '''折線グラフの作成：(x,y)を指定してグラフにプロット'''

    # rouge1のスコアをプロット
    plt.plot(horizontal_axis_list, rouge1_f_score_list, label='rouge1_f_score', color='blue')

    # rouge2のスコアをプロット
    plt.plot(horizontal_axis_list, rouge2_f_score_list, label='rouge2_f_score', color='yellow')

    # rougeLのスコアをプロット
    plt.plot(horizontal_axis_list, rougeL_f_score_list, label='rougeL_f_score', color='green')



    #グラフのタイトルを表示
    plt.title('Scores per Epoch')

    #X軸ラベルを表示
    plt.xlabel('Epoch') 

    #Y軸ラベルを表示
    plt.ylabel('Rouge Score')

    # グラフの表示範囲(y軸方向)
    #plt.ylim(0, 1)

    # グラフの表示範囲(x軸方向)
    plt.xlim(0, len(rouge1_f_score_list))

    #凡例を表示
    plt.legend() 

    # グラフをファイルに保存する
    plt.savefig(result_graph_file_path)


    
# メインモジュールとしてこのファイルを実行する
if __name__ == "__main__":
    
    main()
    