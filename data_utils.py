
from spellchecker import SpellChecker
import torch
import torchvision.transforms as transforms
from PIL import Image
import re
import os.path as path
import torch.nn.functional as F


#計算環境の指定
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')



# dictsファイルがない場合はエラーを出し、ある場合は単語辞書を読み込む
def get_dicts():
    if not path.exists('./weights/dicts'):
        print('ERROR: dicts not created, please run "make dicts"')
        sys.exit(-1)

    dicts = torch.load('./weights/dicts')
    return dicts['stoi'], dicts['itos']



#　ジェネレータの作成関数
def training_dataloader(training_data_number, batch_size, stoi):

    spell = SpellChecker(distance=1)

    #　ミニバッチサイズの返り値になるテンソルをRenNetの入力にあった大きさで定義
    minibatch_images = torch.zeros((batch_size, 3, 224, 224))
    
    # GPUメモリ上にデータを移動
    minibatch_images = minibatch_images.to(device)
    
    # 学習データの画像を格納するリスト
    image_names_for_training = []
    
    
    # 自作したcaptionを読み込む
    with open('./img/official_image_names_for_training.txt') as f:
    ##with open('./img/private_image_names_for_training.txt') as f:

        # .txtの全ての行をリストで取得
        image_names_for_training = [s.strip() for s in f.readlines()]
        
    
    # 全ての学習データの画像を格納するためのテンソルを定義
    images_for_training = torch.zeros((len(image_names_for_training), 3, 224, 224))
    
    
    # 画像名を格納したリストから画像を取得し、その順番にリストに画像をappend→　画像とキャプションは確実に同期される
    #for i in range(len(image_names_for_training)):
    for i in range(training_data_number):
        
        # 学習データの前処理を定義
        tf = transforms.Compose(
                                 [transforms.Resize((224, 224)),
                                 transforms.ToTensor(), 
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        
        # Compose()はImage型データでのみ実行できるので変換→"image_names_for_training.txtから取得する画像名を選択"
        img = Image.open('./img/images_for_training/'+image_names_for_training[i])
        img = tf(img)
        
        images_for_training[i] = img
        
    
    # 自作したcaptionを読み込む
    ##with open('./self_made_captions/traffic_private_train_caption3.txt') as f:
    with open('./self_made_captions/traffic_official_train_caption.txt') as f:
        
        # .txtの全ての行をリストで取得
        self_made_training_caption = [s.strip() for s in f.readlines()]
    
    
    # for文でミニバッチ学習を定義→range関数でミニバッチの個数分ループ
    for batch in range(training_data_number//batch_size):

        # 戻り値として、captionを格納するリストを宣言
        minibatch_captions = []

        
        #　バッチサイズ回数文のループ処理
        for i in range(batch_size):

            # ミニバッチ学習用に作成したテンソルの雛形に画像を格納
            minibatch_images[i] = images_for_training[batch*batch_size+i]

            # キャプションを取得
            target = self_made_training_caption[batch*batch_size+i]

            # フレーズをすべて小文字にし、句読点を区切る
            minibatch_captions.append([word if word in stoi else spell.correction(word) for word in re.findall(r"[\w']+|[.,!?;]", target.lower())])


        # バッチ数分の画像とキャプションを作り終わる度に、yieldで戻り値を返す→　yeildで随時送るので、image_names_for_trainingも調整
        yield minibatch_images, minibatch_captions
            
            

# 検証用の画像/画像キャプションをセットにして返す関数
def validation_dataloader(stoi):
    
    
    # 画像の名前をまとめたテキストファイルを開く
    ##with open('./img/image_names_for_validation.txt') as f:
    with open('./img/official_image_names_for_validation.txt') as f:
        
        # .txtの全ての行をリストで取得
        image_names_for_validation = [s.strip() for s in f.readlines()]
        
    
    # 画像ファイルをこのリストに格納していく
    images_for_validation = []
    
    
    # バリデーション用の画像データを格納したリストを作る
    for i in range(len(image_names_for_validation)):
    
        # 指定したパス上にある画像ファイルを開く
        img = Image.open('img/images_for_validation/'+image_names_for_validation[i])
        
        # 画像の前処理をまとめたtfを定義（ComposeはPIL.Imageにのみ対応）
        tf = transforms.Compose([ transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224, 0.225])])
        
        # 前処理を実行
        img = tf(img)
        
        # 検証用データをリストに格納
        images_for_validation.append(img)
    
    
    # 検証用画像キャプションが書かれているテキストファイルを開く
    ##with open('./self_made_captions/traffic_validation_caption.txt') as f:
    with open('./self_made_captions/traffic_official_validation_caption.txt') as f:

        # .txtの全ての行をリストで取得
        self_made_validation_caption = [s.strip() for s in f.readlines()]
        
        # 検証用の画像キャプションをまとめて格納するリストを作成
        '''captions_for_validation = []
        
        
        # 検証データの正解キャプションをすべて小文字にし、句読点を区切る→　解説用にする
        for i in range(len(self_made_validation_caption)):
            
            # 1つの画像キャプションを選択
            target = self_made_validation_caption[i]
            
            # 読み込んだ画像キャプションを全て小文字に変換
            target = target.lower()
            
            # パターンに一致した部分を抜き出す
            target = re.findall(r"[\w']+|[.,!?;]", target)
            
            # 誤字を最も可能性の高い正しい単語に置き換える
            for word in target:
                
                # 単語辞書に一致する単語があればそのままにする
                if word in stoi:
                    word = word
                
                # そうでない場合は最も可能性の高い、単語辞書ないの単語と置き換える
                else:
                    word = spell.correction(word)
            
            # 画像キャプション毎にリストで囲って区切る
            #caption_for_validation = [target]
             
            # 画像キャプションを返り値になるリストに追加
            #captions_for_validation.append(caption_for_validation)
            captions_for_validation.append(target)'''
        
        
        # 誤字の訂正をするメソッドを持つインスタンスを作成
        spell = SpellChecker(distance=1)
            
        # フレーズをすべて小文字にし、句読点を区切る
        captions_for_validation = [[word if word in stoi else spell.correction(word) for word in re.findall(r"[\w']+|[.,!?;]", self_made_validation_caption[i].lower())] for i in range(len(self_made_validation_caption))]
        
    return images_for_validation, captions_for_validation
            


# テスト用の画像/画像キャプションをセットで返す関数
def test_dataloader():
    
    # テスト画像名のリストを取得
    with open('./img/official_image_names_for_test2.txt', mode='r') as f:
        
        # .txtの全ての行をリストで取得
        image_names_for_test = [s.strip() for s in f.readlines()]


    # テスト画像を格納する型を固定して宣言
    images_for_test = torch.empty(len(image_names_for_test), 3, 224, 224)


    # テスト画像を4次元テンソルに格納していく
    for i in range(len(image_names_for_test)):

        # テスト画像を開く
        img = Image.open('./img/images_for_training/'+image_names_for_test[i])

        # 学習データの前処理を定義
        tf = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # 画像に前処理を行う
        img = tf(img)

        # 画像を4次元テンソルに格納
        images_for_test[i] = img
        
        
    # captionファイルを読み込む
    with open('./self_made_captions/traffic_official_test_caption2.txt') as f:

        # .txtの全ての行をリストで取得
        captions_for_test = [s.strip() for s in f.readlines()]
        
    
    return images_for_test, captions_for_test



# 学習用の画像キャプションデータから、学習データと正解データを作成する関数
def make_train_caption_datasets(stoi, vocabulary, captions):
    
    # 学習キャプションの文頭に文頭記号<s>を追加
    train_labels = [['<s>'] + label for label in captions]
    # ex) train_lables → [[‘<s>’, ‘people’, ‘in’, ‘front’, ‘of’, ‘cars’, ‘.’],…,[…]]
    
    # 正解キャプションの文末に文末記号</s>を追加
    expected_labels = [label + ['</s>'] for label in captions]
    # ex) expected_lables → [[‘people’, ‘in’, ‘front’, ‘of’, ‘cars’, ‘</s>’],…,[…]]
    
    
    # 学習キャプションの単語（文字列型）を単語ID（整数型）に変更
    train_indices = [[stoi[word] for word in label] for label in train_labels]
    # ex) train_indices = [[22635, 150, 200, 10, 30, 22636,…,22636],…,[…]]
    
    # 正解キャプションの単語（文字列型）を単語ID（整数型）に変更
    expected_indices = [[stoi[word] for word in label] for label in expected_labels]
    # ex) expected_indices = [[100, 150, 200, 10, 30, 22636,… ,22636],…,[…]]

    
    # ミニバッチの中で最も多い単語数のキャプションの単語数を変数に保持
    max_length = max([len(label) for label in train_indices])

    
    # 最大単語数より少ない単語数分パディングして全ての単語数を揃える
    train_indices_v = torch.stack([F.pad(torch.tensor(label), pad=(0, max_length-len(label)), mode='constant', value=vocabulary+1) for label in train_indices])
    # ex) train_indices = [[22635, 150, 200, 10, 30, 22636,…,22636],…,[…]]
    
    # 最大単語数より少ない単語数分パディングして全ての単語数を揃える
    expected_indices_v = torch.stack([F.pad(torch.tensor(label), pad=(0, max_length-len(label)), mode='constant', value=vocabulary+1) for label in expected_indices])
    # ex) expected_indices = [[100, 150, 200, 10, 30, 22636,… ,22636],…,[…]]
    
    return train_indices_v, expected_indices_v, train_labels, max_length
