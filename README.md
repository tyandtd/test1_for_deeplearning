DeepLearningMovies
==================

Kaggle's competition for using Google's word2vec package for sentiment analysis

## Installation

There're some requirements for making the stuff work. Use `pip` to install them easily:

```bash
$> sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran
$> sudo pip install -r requirements.txt
```
ps 本篇论文基于
https://www.kaggle.com/c/word2vec-nlp-tutorial/overview/part-1-for-beginners-bag-of-words
其中test1是part1部分
其中test2_model_word2ver是part2部分
其中test2_model_trees是part3部分
增加中文注释，仅为个人理解

其中testdate文件，表示用于测试的文件
traindate，表示用于有标记训练的文件
unlableed，表示为标记的文件
都需要解压后使用
其中300features表示用word2vec训练的模型
