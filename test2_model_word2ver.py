import pandas as pd #用于对文本的基本处理
import nltk.data#用于对文本的基本处理，不导入应该也没问题
# Import various modules for string cleaning
from bs4 import BeautifulSoup#对网页的基本处理
import re
from nltk.corpus import stopwords#导入停用词
# Initialize and train the model (this will take some time)
from gensim.models import word2vec
#导入文本处理的应该库，由goole开发，完成对于大量文本的分析，或者说得到基本的处理方式，词向量
# 该系统非常有意思的一点在于，其采用的训练方式在于预测的方式，即输入当前词的词向量，来预测下一个位置的应该是什么值
# 或者换句话说，其本身不是通过直接训练其词向量（not target），而是一个中间层的参数，最终可以得到每一个单词表中的词向量
# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging#便于压缩文档大小


# Read data from files #读文件，三份
# ps此处是初始的网页下载文件，先两个文件在此处可以不读取
train = pd.read_csv( "labeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )

# Verify the number of reviews that were read (100,000 in total)
# 表明读取完成
print ("Read %d labeled train reviews, %d labeled test reviews, " \
 "and %d unlabeled reviews\n" % (train["review"].size,  
 test["review"].size, unlabeled_train["review"].size ))
print ("\n\n\n\n")

#对文本进行初始处理的函数，
# 其中remove_stopwords表示是否对停用词进行处理
#对于本函数来说，仅是给对下一个文本处理使用
def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML（网页）
    review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters（数字）
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them（小写）
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

#采用nltk中的库，先前已经下载，其目的是拆分文本进入一个句子
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
# 是一个将长文本转化为句子的函数，其目的是问了放于文本文件中进行处理
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            #将其中空的去掉后，放入经过初始处理的句子文本单词进入句子
            # 其本质是将文件由一整条评论分解为了很多句子的的特征
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


sentences = [] # 初始化一个空的句子列表

print ("Parsing Sentences from Training Set\n" )
for review in train["review"]: 
    sentences += review_to_sentences(review, tokenizer) 

print ("Parsing statements from unlabeled set\n\n\n" )
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)
#测试以上代码的运行
print (len(sentences))
# print (sentences[0])
# print (sentences[1])


#这个地方是配置内置的一个日志文件，
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters 
# 设置参数
# ps原本size参数，变为了 vector_size
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words


print ("Training model...\n\n\n\n\n")
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            vector_size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)#为了加快模型的运算

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
#将模型保存，方便后面使用
model_name = "300features_40minwords_10context"
model.save(model_name)

#对于该模型过程一个粗浅的理解，是用大量的句子，有固定长度（此处设为300），然后判断每一个位置是某一个单词的可能性



