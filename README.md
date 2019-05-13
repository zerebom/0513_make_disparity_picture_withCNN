# 0513make_disparity_picture_withCNN

## 概要
CNNを使って、視差画像を生成し、一視点画像から立体視画像を生成する研究。<br>
医療画像に応用し、手術のシミュレーションなどに役立ててもらう。

### 手順
1. 左目・右目用のRGB画像とDisparity mapをそれぞれ用意し、Disparity mapの輝度値をもとに、A視点画像からB視点画像を作成する。<br>
2. 1で生成したB画像とDisparity mapを入力・もともとあるB視点画像を正解データとしてCNN(CAE)の学習を進める

### 課題
- 畳み込みを行い、解像度を下げると出力がぼやける。

### 今後の方針
- 解像度を落とさないAutoEncoderを使って学習を進める。
- 生成する必要がある部分だけ1とした2値行列を入力に入れる。

###　参考
https://github.com/satoshiiizuka/siggraph2017_inpainting
