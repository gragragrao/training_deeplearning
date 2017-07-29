# About

This repository is for reviewing the logic of deep learning and implementing some algorithm using deep learning.(7/23~)

# Books I Used

- 「深層学習」（機械学習プロフェッショナルシリーズ）
- 「ゼロからわかる Deep Learning」(オライリー・ジャパン)
- 「詳解 ディープラーニング」（巣籠悠輔氏著）

# Classes I Took

- AIL Deeplearning基礎講座（東大）
- GCI 消費者インテリジェンス寄与講座（東大）

# Record

面白いと感じた結果などをここに書いていく。


## 活性化関数の比較（deeplearning/activate_function）

シグモイド関数、tanh関数、ReLU関数でそれぞれMNISTのデータ予測を走らせた。

### データセット

- データ数：学習 56000, テスト 14000
- 正規化済み

### 結果

| 関数名 | スコア | かかった時間|
| --- | --- | --- |
| sigmoid | 95.9714285714 | 133.5 s |
| tanh | 96.5928571429 | 132.3 s |
| relu | 98.1071428571 | 135.0 s |
