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

### データセットと学習パラメータ

- データ数：学習 56000, テスト 14000
- 正規化済み

- 重みの初期値：-0.08 < w < 0.08 のランダム値
- 隠れ層：1層（200ユニット）
- 学習係数：0.02（一定）
- バッチ数：10
- 学習エポック：30回

### 結果

| 関数名 | スコア | かかった時間|
| --- | --- | --- |
| sigmoid | 0.956970968899 | 130.9 s |
| tanh | 0.968426349459 | 131.8 s |
| relu | 0.979484172545 | 133.9 s |
