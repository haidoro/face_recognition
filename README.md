# 顔認証

**コードはMac環境のものです。**

## 仕組み

顔認識はHoG特徴量の抽出

SVMでの判別

BーBOX

ResNet　エンコードしたもののL2ノルムの距離で類似度を測る。

顔のランドマークを見つけて登録画像と比較する画像の類似度をみる

### Haar like特徴量

OpenCVで使われているもの

CPUだけでも可能、軽快に動く

顔でない検出がある

正面の顔しか認識できない



### Deep Neural Network

畳み込みニューラルネットワーク

処理が遅い、GPUが必要

### HoG特徴量

顔の部品（HoG）

HoG特徴量とは、Histgrams of Oriented Gradientsで、画像の物体検出で使われる特徴記述子

* セルグリッド上から画像の局所的な輝度と輝度の勾配方向を計算
* 勾配方向を輝度分布のヒストグラム にしたものを特徴量とする
* 画像スケールに対してロバスト

1次微分フィルター



## 顔認証の実装

## virtualenv環境作成

バーチャル環境にvirtualenvを使います。

virtualenvが既にインストールされているか確認

```
virtualenv --version
```

### virtualenvのインストール

バージョン確認してまだインストールされてなければ次の手順でインストールします。既にインストールしていたら、次の「今回の作成アプリの環境設定」に進みます。

virtualenvのインストールコマンド

```
sudo pip install virtualenv
```



### 今回作成アプリの環境設定

flask_vggという環境を作るには、以下を実行

```
virtualenv face_recognition
```



環境に入るのは以下`cd face_recognitionで移動した後に次のコマンドを実行

ここはよく失敗するので注意。

#### Mac環境での仮想環境に入り方

要は`virtualenv`で作成した環境名のフォルダ内にbinフォルダが作成されます。binフォルダ内にactivateファイルがあるのでそれを実行すれば良いのです。

```python
source bin/activate
```

現在の仮想環境から出るには次のコード。

```python
deactivate
```

### Windows環境での仮想環境に入り方

```python
Scripts\activate
```

現在の仮想環境から出るには次のコード。

```python
Scripts\deactiv
```



### ライブラリのインストール



ライブラリインストールコマンドはpipを使います。

```
pip install opencv-contrib-python
```

```
pip install cmake
```

```
pip install face_recognition
```

#### Windowsではface_recognitionのインストールに失敗します

2020年5月現在

Windowsの場合、face_recognitionのインストールする前に、次の環境を作る必要があります。

Visual Studio 2019のページの「Visual Studio のダウンロード」ボタンから「vs_buildtools__.exe」ファイルをダウンロードします。（\*部分に番号が入っています。）



ダウンロードしたファイルをインストールすると、Visual Studio Installerが使えるようになります。

![](https://itstudio.co/sample/images/vsinst.jpg)

[変更]を選ぶと次のようなメニューがあります。

この中のC++ Build Toolsにチェックを入れて実行すると環境が出来上がります。

![](https://itstudio.co/sample/images/vsinst2.jpg)



その後次のインストールを行うとうまくいきます。

```
pip install face_recognition
```





### 画像の準備

imagesフォルダを準備して認証したい顔写真（自分の正面写真）を登

### 顔認証のコード



show_name.pyコード

```
# -*- coding: utf-8 -*-
import sys
import face_recognition
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import glob

threshold = 0.5

"""顔情報の初期化"""

face_locations = []
face_encodings = []

"""登録画像の読み込み"""

image_paths = glob.glob('image/*')
image_paths.sort()
known_face_encodings = []
known_face_names = []
checked_face = []

delimiter = "/" # Mac / Linux用

for image_path in image_paths:
    im_name = image_path.split(delimiter)[-1].split('.')[0]
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(im_name)

video_capture = cv2.VideoCapture(0)

def main():

    while True:
        # ビデオの単一フレームを取得
        _, frame = video_capture.read()

        # 顔の位置情報を検索
        face_locations = face_recognition.face_locations(frame)
        # 顔画像の符号化
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding in face_encodings:
            # 顔画像が登録画像と一致しているか検証
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, threshold)
            name = "Unknown"

            # 顔画像と最も近い登録画像を候補とする
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # 位置情報の表示
        for (top, right, bottom, left) in face_locations:
            # 顔領域に枠を描画
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # 顔領域の下に枠を表示
            cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # 結果をビデオに表示
        cv2.imshow('Video', frame)

        # ESCキーで終了
        if cv2.waitKey(1) == 27:
            break

main()

video_capture.release()
cv2.destroyAllWindows()
```

### 顔認証の実行

MACのノートに付いているカメラで自分の動画を撮影し、リアルタイムに認証します。

仮想環境を実行した上で行います。

**実行するには次のコマンド**

```
python show_name.py
```

動画の画面になります。

自分の顔に四角い領域ができて、下部に名前が表示されます。

終了はescキー

## 通常の操作方法

以降、開始するには、cdコマンドでface_recognitionフォルダに移動して、次の2つのコマンドで実行できます。



**仮想環境の開始コマンド**

```python
source bin/activate
```

**キャプチャー実行コマンド**

```
python show_name.py
```

**キャプチャー終了**

**escキー**

**現在の仮想環境から出るコマンド**

```python
deactivate
```

## 