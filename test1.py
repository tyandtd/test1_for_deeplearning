# 导入 pandas 包，然后使用“read_csv”函数读取
# 标记的训练数据
import pandas as pd      
from bs4 import BeautifulSoup   
import re
import nltk 
import numpy as np 
from nltk.corpus import stopwords # 导入停用词列表
train = pd.read_csv("labeledTrainData.tsv", header=0, \
                        delimiter="\t", quoting=3)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
#这个地方是对于一个例子的测试
#example1 = BeautifulSoup(train["review"][0])   #作用是删除标签，就是网页排版中的一些属性？
# Use regular expressions to do a find-and-replace
#letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for,put not a-zA-Z
#                      " ",                   # The pattern to replace it with ,to ' '
#                      example1.get_text() )  # The text to search 数字删除
#lower_case = letters_only.lower()        # Convert to lower case大小写转换
#words = lower_case.split()               # Split into words单词分离
#words = [w for w in words if not w in stopwords.words("english")]
#print (words)
#以上是单个数据的导入，下面是对数据处理定义的函数
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   
# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size#得到总数

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in range( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    if( (i+1) %1000 == 0 ): 
        print ("评论 %d of %d\n" % ( i+1, num_reviews ))
    clean_train_reviews.append( review_to_words( train["review"][i] ) )
    
print ("pre_set is over\n\n\n\n")
print ("Creating the bag of words...\n")
#选择出5000个单词（最常见）
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

#print (train_data_features.shape)#输出行列的个数

vocab = vectorizer.get_feature_names_out() 
#print (vocab)#展示选出词汇用

# Sum up the counts of each vocabulary word
#计数用
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
#展示选出词汇用，即对应的词
# for tag, count in zip(vocab, dist):
#     print (count, tag)
##生成随机森林
print ("\n\n\n\nTraining the random forest...")
# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["sentiment"] )

# Read the test data
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", \
                   quoting=3 )

# Verify that there are 25,000 rows and 2 columns
print ("\n\n\n\n")
print (test.shape)

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = [] 
#使用相似的方式处理数据
print ("Cleaning and parsing the test set movie reviews...\n")


for i in range(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print ("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
#然后把预测的数据放入，预测函数中
result = forest.predict(test_data_features)
print ("\n\n\n\n")
print (sum(round(np.add(result,test.sentiment)%2))/25000.0)
print ("\n\n\n\n")
# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )

