# LSTM, 夏目漱石, 予測
"""
import 
  - sys
  - tensorflow
  - keras
  - numpy
  - os

"""
import os
import sys
import numpy as np

from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Activation, LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint

class text_generation:
    # temperatureがdiversityに対応
    def sample(self, preds, temperature=1.0):
        # numpyで用いられるarray形式に変換
        preds = np.asarray(preds).astype("float64") 

        # 1 / temperature * log(preds) を計算  log(preds ** (1 / temperature)) 
        # temperatureは平滑化のための係数で、値が大きいほど、確率分布が平になる (つまり言語モデルによる確率分布が重要視されなくなる)
        preds = np.log(preds) / temperature       

        # exp_preds = preds ** (1 / temperature)を計算　expを取るのは確率の定義である値が0以上であるようにするため
        exp_preds = np.exp(preds)   

        # predsを確率分布として計算
        preds = exp_preds / np.sum(exp_preds)       

        # predsに従って文字を一つサンプリング　probsはサンプリングされた文字インデックスのみ1、それ以外は0となるarrayとなる
        probs = np.random.multinomial(1, preds, 1)   

        # argmaxを取ることによって、サンプリングされた文字インデックス
        return np.argmax(probs)

    # 100語の文字列(gerated) を返す
    def generation(self, message_text = "君の名前に"):
        # 訓練用データをGoogleDriveから取得
        path = "./source/train_text.txt"
        bindata = open(path, "rb").read()
        text = bindata.decode("utf-8")


        # テキストの 文字数 と 文字種類 を出力
        print("Size of text: ",len(text))   # 160304
        chars = sorted(list(set(text)))
        print("Total chars :",len(chars))   # 2063


        # 学習データ(50文字)と検証データ(次の1文字)のリストを作成
        maxlen = 3   # 学習データ文字数
        step = 3   # 学習データの間隔
        sentences = []  # 学習データ
        next_chars = [] # 検証データ
        epochs = 80


        # 3文字（step）ずらしながら、学習データと検証データのリストを作成
        for i in range(0, len(text)-maxlen, step): 
            # 50文字リストに追加
            sentences.append(text[i: i+maxlen])   # i〜i+50
            # 次の1文字リストに追加
            next_chars.append(text[i+maxlen]) 

        print("size of sentences : ", len(sentences))   # 学習データの長さ   # 53418
        print("size of next_chars : ", len(next_chars))   # 検証データの長さ   # 53418


        #辞書を作成
        char_indices = dict((c,i) for i,c in enumerate(chars)) # {"文字1": 番号1, "文字2": 番号2, ・・・ }   # {'―': 0, '…': 1, '、': 2, '。': 3, '々': 4, '「': 5, '」': ・・・ }
        indices_char = dict((i,c) for i,c in enumerate(chars)) # {番号1: "文字1", 番号2: "文字2", ・・・ }   # {0: '―', 1: '…', 2: '、', 3: '。', 4: '々', 5: '「', ・・・ }


        # テキストのベクトル化
        # 配列の全要素をFalseで初期化
        x = np.zeros((len(sentences),maxlen,len(chars)),dtype=np.bool)
        y = np.zeros((len(sentences),len(chars)),dtype=np.bool)

        # 番号(i) と 学習データ(sentence"s") のループ
        for i, sentence in enumerate(sentences): 
            # 番号(i) と 50文字(sentence) のループ
            for t ,char in enumerate(sentence): 
                x[i,t,char_indices[char]] = 1 # char_indices[char] = 文字に該当する辞書char_indicesの番号
            y[i,char_indices[next_chars[i]]] = 1 # 次の1文字に該当する辞書char_indicesの番号

        print("x.shape : ", x.shape)   # 学習データのベクトル の大きさ   #  (53418, 50, 2063)
        print("y.shape : ", y.shape)   # 検証データのベクトル の大きさ   #  (53418, 2063)


        # モデルを読み込む
        model = model_from_json(open('./models/model_maxlen_' + str(maxlen) + '_epoch_' + str(epochs) + '.json').read())
        # 学習結果を読み込む
        model.load_weights('./models/model_maxlen_' + str(maxlen) + '_epoch_' + str(epochs) + '.h5')

        input_word = message_text

        for diversity in [0.5]:
            print()
            print("-----diversity", diversity)

            generated = ""

            start_index = 0
            # 40文字の文章
            #sentence = text[start_index: start_index + maxlen ]
            sentence = input_word
            generated += sentence
            print("-----Seedを生成しました: ")
            print(sentence)
            
            sys.stdout.write(generated)

            # 40文字のあとの、次の400文字を予測
            for i in range(100):

                # sentenceのベクトル化
                x = np.zeros((1,maxlen,len(chars)))
                for t,char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1

                # 次の文字を予測
                preds = model.predict(x, verbose =9)[0]
                # 次の1文字のインデックス
                next_index = self.sample(preds, diversity)
                # 次の1文字
                next_char = indices_char[next_index]
                # 次の1文字を追加する
                generated += next_char

                # 2文字目からに、次の1文字を追加した40字（1文字ずらした40字）
                sentence = sentence[1:] + next_char
                
                # 次の1文字を追記。コンソールに出力
                sys.stdout.write(next_char)
                sys.stdout.flush()
            #print(generated)
            print() 

        return generated
