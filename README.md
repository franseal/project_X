![create_ex](https://user-images.githubusercontent.com/46518704/50991432-f77ccc80-1557-11e9-95b0-2ae3ddb14778.jpg)

# project_X
プロジェクト実習Xで作成しました。  
5-7-5の形の川柳みたいなやつを生成するプログラムです。

# できること
小説等の文章を利用してLSTMで学習し、17拍(5-7-5)の形の川柳のようなものを生成できます。  
川柳の表現技法(皮肉の表現など)は考慮せず、ひたすら17拍の川柳をたくさん生成します。  
明らかに川柳じゃないものが大量に生成されますが、稀に本物っぽい川柳が生成されたりもします。  

# 開発・実行環境
Google Colaboratory(Python 3)  
janome 0.3.7  
mecab-python3-0.996.1  
chainer 5.0.0  
cupy 5.0.0  

# 使い方
### 用意するもの
・学習データとなる小説等文章のテキストファイル(.txt)  
**句点で区切って学習用データを作成しています。  
句点が含まれているテキストファイルを用意してください。**  
```
!wget http://www.cl.ecei.tohoku.ac.jp/nlp100/data/neko.txt
```  
の実行で「吾輩は猫である」の本文が取得できます。  
または  
```
from google.colab import files
uploaded = files.upload()
```  
の実行でローカルからテキストファイルをアップロードできます。  
***
### 環境の準備
基本的にはipynbファイル内の「準備」部分のセルを実行すれば完了します。  
```
!wget http://cl.ecei.tohoku.ac.jp/nlp100/data/neko.txt
```  
の部分はローカルからテキストファイルをアップロードする場合は実行してなくても問題ありません。  
***
### テキストの分かち書き
「分かち書き」部分にてテキストファイルの分かち書きを行います。  
このセクションで学習用データの作成まで行います。  
ローカルからテキストファイルをアップロードする際は
```
from google.colab import files
uploaded = files.upload()
```  
のセルを実行してファイルをアップロードしてください。

```
text = open("neko.txt").readlines()
```  
の```"neko.txt"```部分をアップロードしたファイル名に変更してください。

以上が完了したら  
```
wakati = []
for t in tqdm(text):
    for row in mecab.parse(t).split("\n"):
    ...
```  
のセルから順に実行してください。
***
### ネットワーク
LSTMを作成する部分です。  
3つのセルを順番に実行するだけで完了です。
***
### 学習
```
from tqdm import trange

batchsize = 128
for epoch in range(30): #学習回数
  shuffled = np.random.permutation(len(train_data))
  sum_loss = 0.0
  n=0
  
  for i in trange(0, len(train_data), batchsize):
    ids = shuffled[i:i+batchsize]
    xs = [cp.array(train_data[i]) for i in ids]
    
    model.cleargrads()
    loss = model(xs)
    loss.backward()
    optimizer.update()
    
    sum_loss += loss.data
    #print(loss.data)
    n += len(ids)
    
  print("Epoch {} : loss {}".format(epoch, sum_loss / n))
```  
`for epoch in range(param1):`

+ `param1` :  
学習回数を設定します。  

このセルを実行することで学習が開始されます。
***
### 生成
まず最初に  
```
from janome.tokenizer import Tokenizer
import re
...
```  
のセルを実行しておきます。  
次に`def first()`、`def second()`、`def third()`を実行します。
***

```def first(param1, param2):```  
上五の生成を行う関数です。  

+ `param1` :  
文字列をID変換した配列が代入されます。  
+ `param2` :  
モデルの推測結果のうち、確率の高いものから順にいくつまで推測結果を利用するか設定します。  
***

```def second(param1, param2, param3):```  
中七の生成を行う関数です。  
```def third(param1, param2, param3):```  
下五の生成を行う関数です。  

+ `param1` :  
文字列をID変換した配列が代入されます。  
+ `param2` :  
モデルの推測結果のうち、確率の高いものから順にいくつまで推測結果を利用するか設定します。  
+ `param3` :  
確率の高いものを選択した後、何回まで推測・生成を行うか設定します。  
***

```
f_rank = param1 #上五の上位数
s_rank = param2 #中七の上位数
s_score = param3 #中七の生成数
t_rank = param4 #下五の上位数
t_create = param5 #下五の生成数

sen = [word2id["param6"]]
```  
上五・中七・下五を生成する関数に渡す引数の設定を行います。  
同時に川柳を生成する際に一番最初にくる単語の設定を行います。  

+ `param1` :  
`def first()`の`param2`に渡されます。  
+ `param2` :  
`def second()`の`param2`に渡されます。  
+ `param3` :  
`def second()`の`param3`に渡されます。  
+ `param4` :  
`def third()`の`param2`に渡されます。  
+ `param5` :  
`def third()`の`param3`に渡されます。  
+ `param6` :  
川柳生成を行う際に使用する一番初めの単語を設定します。  
**学習に使用したテキストファイル中に出現する単語のみ設定できます。  
テキストファイル中に出現しない単語を入力した場合はエラーになります。**
***

```
kari = []
res2 = []
res3 = []

res = first(sen, f_rank)
print(res)

for i in range(len(res)):
  sen = res[i]
  kari = second(sen, s_rank, s_create)
  for j in range(len(kari)):
    res2.append(kari[j])
print(res2)

for i in range(len(res2)):
  sen = res2[i]
  kari = third(sen, t_rank, t_create)
  for j in range(len(kari)):
    res3.append(kari[j])
print(res3)

for i in range(len(res3)):
  s = "".join([id2word[i] for i in res3[i]])
  print(s)
```  
このセルを実行すると川柳が生成されます。
