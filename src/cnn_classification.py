import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Merge
from keras.layers import *
from keras.regularizers import *
import sys
from gensim.models.word2vec import Word2Vec
import gensim
import keras

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

GLOVE_DIR = "../model/"
MAX_SEQUENCE_LENGTH = 40
EMBEDDING_DIM = 300
MAX_NB_WORDS = 20000 # 整体词库字典中，词的多少，可以略微调大或调小

def loadWord2Vec():

    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt')) # 读入50维的词向量文件，可以改成100维或者其他
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def loadWord2VecGoogle():

    embeddings_index = {}

    vector_path = '/home/yaoyang/GoogleNews-vectors-negative300.bin'
    model = gensim.models.KeyedVectors.load_word2vec_format(vector_path, binary=True)

    for word in model.vocab:
        embeddings_index[word] = model[word]

    return embeddings_index


def textProcessing(textPath, labelPath):

    print('Processing text dataset')
    texts = []  # 存储训练样本的list
    labels = []

    textFile = open(textPath, encoding='latin-1')
    labelFile = open(labelPath, encoding='latin-1')
    lines = textFile.readlines()  # 读取全部内容
    for line in lines:
        texts.append(line)
    lines = labelFile.readlines()
    for line in lines:
        labels.append(int(line)-1)

    print("sample num:", len(texts))
    print("label num:", len(labels))
    return texts, labels

def tokenEncoding(trainText, trainLabel, validText, validLabel):

    totalText = trainText+validText

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(totalText)
    trainSequences = tokenizer.texts_to_sequences(trainText)
    validSequences = tokenizer.texts_to_sequences(validText)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    trainData = pad_sequences(trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    validData = pad_sequences(validSequences, maxlen=MAX_SEQUENCE_LENGTH)

    trainLabelVec = to_categorical(np.asarray(trainLabel))
    valLabelVec = to_categorical(np.asarray(validLabel))

    print('Shape of train data tensor:', trainData.shape)
    print('Shape of train label tensor:', trainLabelVec.shape)

    print('Shape of valid data tensor:', validData.shape)
    print('Shape of valid label tensor:', valLabelVec.shape)

    return trainData, trainLabelVec, validData, valLabelVec, word_index

def trainCNN(x_train, y_train, x_val, y_val, word_index, embeddings_index):

    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector  # word_index to word_embedding_vector ,<20000(nb_words)

    # load pre-trained word embeddings into an Embedding layer
    # 神经网路的第一层，词向量层，本文使用了预训练glove词向量，可以把trainable那里设为False
    embedding_layer = Embedding(nb_words + 1,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH,
                                weights=[embedding_matrix],
                                trainable=False)

    print('Training model.')


    # train a 1D convnet with global maxpoolinnb_wordsg

    # left model 第一块神经网络，卷积窗口是5*dim（dim是词向量维度）
    model_left = Sequential()
    # model.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
    model_left.add(embedding_layer)
    model_left.add(Conv1D(128, 5, padding="SAME", activation='relu'))
    model_left.add(MaxPooling1D(2))
    model_left.add(Conv1D(128, 5, padding="SAME", activation='relu'))
    model_left.add(MaxPooling1D(2))
    model_left.add(Conv1D(128, 5, padding="SAME", activation='relu'))
    model_left.add(MaxPooling1D(10))
    model_left.add(Flatten())

    print(model_left.summary())

    # right model <span style="font-family: Arial, Helvetica, sans-serif;">第二块神经网络，卷积窗口是4*50</span>

    model_right = Sequential()
    model_right.add(embedding_layer)
    model_right.add(Conv1D(128, 4, padding="SAME",activation='relu'))
    model_right.add(MaxPooling1D(2))
    model_right.add(Conv1D(128, 4, padding="SAME", activation='relu'))
    model_right.add(MaxPooling1D(2))
    model_right.add(Conv1D(128, 4, padding="SAME", activation='relu'))
    model_right.add(MaxPooling1D(10))
    model_right.add(Flatten())

    # third model <span style="font-family: Arial, Helvetica, sans-serif;">第三块神经网络，卷积窗口是6*50</span>
    model_3 = Sequential()
    model_3.add(embedding_layer)
    model_3.add(Conv1D(128, 3, padding="SAME", activation='relu'))
    model_3.add(MaxPooling1D(2))
    model_3.add(Conv1D(128, 3, padding="SAME", activation='relu'))
    model_3.add(MaxPooling1D(2))
    model_3.add(Conv1D(128, 3, padding="SAME", activation='relu'))
    model_3.add(MaxPooling1D(10))
    model_3.add(Flatten())

    merged = Merge([model_left, model_right, model_3],
                   mode='concat')  # 将三种不同卷积窗口的卷积层组合 连接在一起，当然也可以只是用三个model中的一个，一样可以得到不错的效果，只是本文采用论文中的结构设计
    model = Sequential()
    model.add(merged)  # add merge
    model.add(Dropout(0.5))
    #model.add(Dense(128, activation='tanh'))  # 全连接层
    model.add(Dense(5, activation='softmax'))  # softmax，输出文本属于20种类别中每个类别的概率

    opt = keras.optimizers.Adadelta(lr=0.5, rho=0.95, epsilon=1e-06)

    # 优化器我这里用了adadelta，也可以使用其他方法
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    print(model.summary())

    # =下面开始训练，nb_epoch是迭代次数，可以高一些，训练效果会更好，但是训练会变慢
    model.fit(x_train, y_train, batch_size=48, nb_epoch=15,  validation_data=(x_val, y_val))

    score = model.evaluate(x_train, y_train, verbose=0)  # 评估模型在训练集中的效果，准确率约99%
    print('train score:', score[0])
    print('train accuracy:', score[1])
    score = model.evaluate(x_val, y_val, verbose=0)  # 评估模型在测试集中的效果，准确率约为97%，迭代次数多了，会进一步提升
    print('Test score:', score[0])
    print('Test accuracy:', score[1])



if __name__ == '__main__':

    trainxPath = "../data/train_x.txt"
    trainyPath = "../data/train_y.txt"
    valxPath = "../data/dev_x.txt"
    valyPath = "../data/dev_y.txt"

    #embeddings_index = loadWord2Vec()
    embeddings_google = loadWord2VecGoogle()

    trainTexts, trainLabels = textProcessing(trainxPath , trainyPath)

    valTexts, valLabels = textProcessing(valxPath, valyPath)
    trainx, trainy, valx, valy, wordIndex = tokenEncoding(trainTexts, trainLabels, valTexts, valLabels)

    trainCNN(trainx, trainy, valx, valy, word_index=wordIndex, embeddings_index=embeddings_google)
