#ps其中的syn0变成了vectors，原本的model变成了model.wv
import numpy as np  # Make sure that numpy is imported#常用的一个头文件，用于数组或者说矩阵的处理
from gensim.models import Word2Vec#goole推出的一个用于文本分析的软件，在本程序中的主要目的，下载上文已经生成好的模型
from sklearn.cluster import KMeans #导入k—mean聚类模型
import time#计时
import pandas as pd#用于对文本的预处理
from sklearn.ensemble import RandomForestClassifier#导入随机森林模型
# Set values for various parameters#这个地方，是上一步程序完成的对于数据完成的一个文本学习后的一个过程
model = Word2Vec.load("300features_40minwords_10context")#导入模型
# print (type(model.wv.vectors))

print (model.wv["flower"][2])
#print (np.array(model.wv.index_to_key).shape)
#在这个地方验证的结果表明：我们拥有（16490，300）个数据



# num_features = 300    # Word vector dimensionality                      
# min_word_count = 40   # Minimum word count                        
# num_workers = 4       # Number of threads to run in parallel
# context = 10          # Context window size                                                                                    
# downsampling = 1e-3   # Downsample setting for frequent words


# # Read data from files #读取爬虫下载的网页文件
# # ps在此处是不用读取最后的unlabeled_train的数据的，那一份数据应该用于在使用在建立模型过程中
# train = pd.read_csv( "labeledTrainData.tsv", header=0, 
#  delimiter="\t", quoting=3 )

# test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )

# unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, 
#  delimiter="\t", quoting=3 )

# # Verify the number of reviews that were read (100,000 in total)，这个地方，是为了表明读取完成
# print ("Read %d labeled train reviews, %d labeled test reviews, " \
#  "and %d unlabeled reviews\n" % (train["review"].size,  
#  test["review"].size, unlabeled_train["review"].size ))
# ######################################################################################
# ######################################################################################
# #####################################数据处理#########################################
# ######################################################################################
# ######################################################################################
# from bs4 import BeautifulSoup#用于对于文本处理，于上出基本一致
# import re
# #导入停用词
# from nltk.corpus import stopwords
# import nltk.data 
# #导入划分句子的库
# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') 
# #
# def review_to_wordlist( review, remove_stopwords=False ):
#     # Function to convert a document to a sequence of words,
#     # optionally removing stop words.  Returns a list of words.
#     #
#     # 1. Remove HTML
#     review_text = BeautifulSoup(review).get_text()
#     #  
#     # 2. Remove non-letters
#     review_text = re.sub("[^a-zA-Z]"," ", review_text)
#     #
#     # 3. Convert words to lower case and split them
#     words = review_text.lower().split()
#     #
#     # 4. Optionally remove stop words (false by default)
#     if remove_stopwords:
#         stops = set(stopwords.words("english"))
#         words = [w for w in words if not w in stops]
#     #
#     # 5. Return a list of words
#     return(words)

# # Define a function to split a review into parsed sentences
# #其实不一定用的上
# def review_to_sentences( review, tokenizer, remove_stopwords=False ):
#     # Function to split a review into parsed sentences. Returns a 
#     # list of sentences, where each sentence is a list of words
#     #
#     # 1. Use the NLTK tokenizer to split the paragraph into sentences
#     raw_sentences = tokenizer.tokenize(review.strip())
#     #
#     # 2. Loop over each sentence
#     sentences = []
#     for raw_sentence in raw_sentences:
#         # If a sentence is empty, skip it
#         if len(raw_sentence) > 0:
#             # Otherwise, call review_to_wordlist to get a list of words
#             sentences.append( review_to_wordlist( raw_sentence, \
#               remove_stopwords ))
#     #
#     # Return the list of sentences (each sentence is a list of words,
#     # so this returns a list of lists
#     return sentences

# #对句子的处理，该函数用不上
# # sentences = []  # Initialize an empty list of sentences

# # print ("Parsing sentences from training set")
# # for review in train["review"]:
# #     sentences += review_to_sentences(review, tokenizer)

# ##处理数据，训练和标记
# clean_train_reviews = []
# for review in train["review"]:
#     clean_train_reviews.append( review_to_wordlist( review, \
#         remove_stopwords=True ))
    


# print ("处理数据，训练和标记")
# clean_test_reviews = []
# for review in test["review"]:
#     clean_test_reviews.append( review_to_wordlist( review, \
#         remove_stopwords=True ))

# ######################################################################################
# ######################################################################################
# #####################################v1###############################################
# ######################################################################################
# ######################################################################################
# #每出现一个单词，就将他在各个位置可能会出现的可能性，全部加在一起，最后仅输入300个特征值
# #ps关于后面对于标记文字，为什么要进行不考虑文字会出现的位置？因为标记文字会进行stop word的删除，其本身
# def makeFeatureVec(words, model, num_features):
#     # Function to average all of the word vectors in a given
#     # paragraph
#     #预分配
#     # Pre-initialize an empty numpy array (for speed)
#     featureVec = np.zeros((num_features,),dtype="float32")
#     #
#     nwords = 0.
#     # 
#     # Index2word is a list that contains the names of the words in 
#     # the model's vocabulary. Convert it to a set, for speed 
#     index2word_set = set(model.index_to_key)
#     #
#     # Loop over each word in the review and, if it is in the model's
#     # vocaublary, add its feature vector to the total
#     for word in words:
#         if word in index2word_set: 
#             nwords = nwords + 1.
#             featureVec = np.add(featureVec,model[word])
#     # 
#     # Divide the result by the number of words to get the average
#     #归一化处理
#     featureVec = np.divide(featureVec,nwords)
#     return featureVec



#        #此处为重复执行过程一，本身仅仅是一个将数据统一处理的函数
# def getAvgFeatureVecs(reviews, model, num_features):
#     # Given a set of reviews (each one a list of words), calculate 
#     # the average feature vector for each one and return a 2D numpy array 
#     # 
#     # Initialize a counter
#     counter = 0.
#     # 预分配空间
#     # Preallocate a 2D numpy array, for speed
#     reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
#     # 千次计数
#     # Loop through the reviews
#     for review in reviews:
#        #
#        # Print a status message every 1000th review
#        if counter%1000. == 0.:
#            print ("Review %d of %d" % (counter, len(reviews)))
#        # 
#        # Call the function (defined above) that makes average feature vectors
#        #ps一点要记住，这个地方加入round，保证其为整数
#        reviewFeatureVecs[round(counter)] = makeFeatureVec(review, model, \
#            num_features)
#        #
#        # Increment the counter
#        counter = counter + 1.
#     return reviewFeatureVecs


# # ****************************************************************
# # Calculate average feature vectors for training and testing sets,
# # using the functions we defined above. Notice that we now use stop word
# # removal.
# #将数据变为需要的样子，分别变化两者
# trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model.wv, num_features )
# print ("Creating average feature vecs for test reviews")
# testDataVecs = getAvgFeatureVecs( clean_test_reviews, model.wv, num_features )



# #################################建立100颗树进行随机深林，一个较为简单的重复过程

# # Fit a random forest to the training data, using 100 trees
# forest = RandomForestClassifier( n_estimators = 100 )

# print ("Fitting a random forest to labeled training data...")
# forest = forest.fit( trainDataVecs, train["sentiment"] )

# # Test & extract results 
# result = forest.predict( testDataVecs )
# print ("\n\n\n\n")
# print (sum(result)/25000.0)
# print ("\n\n\n\n")
# # Write the test results 
# output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
# output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )



# #####################################################################################
# #####################################################################################
# ####################################v2###############################################
# #####################################################################################
# #####################################################################################

# start = time.time() # Start time标记开始时间
# # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# # average of 5 words per cluster
# word_vectors = model.wv.vectors
# #得到所有的单词变成的集合，及其对应位置的值
# num_clusters = round(word_vectors.shape[0] / 5)
# #将单词总数的1/5作为聚类的个数，这个地方要将其整数化，否则函数不认可
# #print (num_clusters)
# # Initalize a k-means object and use it to extract centroids
# kmeans_clustering = KMeans( n_clusters = num_clusters )
# #聚类的个数
# idx = kmeans_clustering.fit_predict( word_vectors )
# #进行k聚类的过程，进行k聚类的具体原因是为了更好
# # Get the end time and print how long the process took
# end = time.time()
# elapsed = end - start
# print ("Time taken for K Means clustering: ", elapsed, "seconds.")


# # Create a Word / Index dictionary, mapping each vocabulary word to
# # a cluster number 
# #生成一个dict的单词序列                                                                                           
# word_centroid_map = dict(zip( model.wv.key_to_index, idx ))

# #测试代码，其中value表示单词在哪个集群中间，key表示单词
# # # For the first 10 clusters
# # for cluster in range(0,10):
# #     #
# #     # Print the cluster number  
# #     print ("\nCluster %d" % cluster)
# #     #
# #     # Find all of the words for that cluster number, and print them out
# #     words = []
# #     for i in range(0,len(word_centroid_map.values())):
# #         if( list(word_centroid_map.values())[i] == cluster ):
# #             words.append(list(word_centroid_map.keys())[i])
# #     print (words)
# # print (word_centroid_map)
# # print (word_centroid_map.shape)


# #如果出现第i个单词，在表格中搜索该单词，然后将第i个集群位置的量加一，得到每一个集群出现的次数，作为输入变量
# #ps我感觉到非常有意思的一点，其先后顺序的概率，在最后被溶解到了各个集群中间，但是在新加入时，就很少再考虑，新加入的单词所应该在的位置

# def create_bag_of_centroids( wordlist, word_centroid_map ):
#     #
#     # The number of clusters is equal to the highest cluster index
#     # in the word / centroid map
#     num_centroids = max( word_centroid_map.values() ) + 1
#     #
#     # Pre-allocate the bag of centroids vector (for speed)
#     bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
#     #
#     # Loop over the words in the review. If the word is in the vocabulary,
#     # find which cluster it belongs to, and increment that cluster count 
#     # by one
#     for word in wordlist:
#         if word in word_centroid_map:
#             index = word_centroid_map[word]
#             bag_of_centroids[index] += 1
#     #
#     # Return the "bag of centroids"
#     return bag_of_centroids   



# #预分配空间
# # Pre-allocate an array for the training set bags of centroids (for speed)
# train_centroids = np.zeros( (train["review"].size, num_clusters), \
#     dtype="float32" )
# #将要训练的值，变化成为，要求输入的值，即各个集群出现的次数
# # Transform the training set reviews into bags of centroids
# counter = 0
# for review in clean_train_reviews:
#     train_centroids[counter] = create_bag_of_centroids( review, \
#         word_centroid_map )
#     counter += 1
# #同样于上步操作
# # Repeat for test reviews 
# test_centroids = np.zeros(( test["review"].size, num_clusters), \
#     dtype="float32" )

# counter = 0
# for review in clean_test_reviews:
#     test_centroids[counter] = create_bag_of_centroids( review, \
#         word_centroid_map )
#     counter += 1
# #训练决策树的模型
# # Fit a random forest and extract predictions 
# forest = RandomForestClassifier(n_estimators = 100)
# # Fitting the forest may take a few minutes
# print ("Fitting a random forest to labeled training data...")
# forest = forest.fit(train_centroids,train["sentiment"])
# result2 = forest.predict(test_centroids)
# # Write the test results 

# output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
# output.to_csv( "BagOfCentroids.csv", index=False, quoting=3 )
# print (sum( (np.add(result,result2)%2)))


