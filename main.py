import json
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import save_model
from sklearn.metrics import fbeta_score
import random
import pickle
import qalsadi.lemmatizer
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import pandas as pd
import tflearn

df = pd.read_csv("data.csv")

with open("response.json",'r', encoding='utf-8') as file:
    data = json.load(file)
try:
    with open('data.pickle','rb') as f:
        words, classes, training, output_row, output = pickle.load(f)
except:
    lemmer = qalsadi.lemmatizer.Lemmatizer()
    tokenized_words=[]
    classes = []
    doc = []
    ignoring_words = [ '?','؟' ,',', '.','!']

    for i in range (df['message'].count()):
        word = nltk.word_tokenize(df['message'][i])
        tokenized_words.extend(word)
        doc.append((word, df['intent'][i]))
        if df['intent'][i] not in classes:
            classes.append(df['intent'][i])
    
    classes = sorted(list(set(classes)))
    tokenized_words = [stemmer.stem(w.lower()) for w in tokenized_words if w not in ignoring_words ]
    tokenized_words = sorted(list(set(tokenized_words)))
    training = []
    output = []
    out_empty = [0 for _ in range(len(classes))]

    for x, docs in enumerate(doc):
        bag = []

        wrds =docs[0]

        for w in tokenized_words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
                
        output_row = out_empty[:]
        output_row[classes.index(docs[1])] = 1 
        training.append(bag)
        output.append(output_row)
    training = np.array(training)
    output = np.array(output)     

    with open("data.pickle", "wb") as f:
        pickle.dump((tokenized_words, classes, training, output), f)



from tensorflow.python.framework import ops
ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(sentence, words):
    bag = [0] * len(words)
    s_words = nltk.word_tokenize(sentence)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)


def predict_class(sentence, words):
    # filter out predictions below a threshold
    p =  bag_of_words(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list = classes[r[0]]
    return return_list

#
#
#peer grouping
#
#
group_A1 = "https://discordapp.com/channels/999450036620492870/999666944032637029"

group_A2 = "https://discordapp.com/channels/999450036620492870/999667011242172476"

group_B1 = "https://discordapp.com/channels/999450036620492870/999667053176832001"

group_B2 = "https://discordapp.com/channels/999450036620492870/999667117131575306"

group_C1 = "https://discordapp.com/channels/999450036620492870/999667173117153420"

group_C2 = "https://discordapp.com/channels/999450036620492870/999667229299847278"

def peer_grouping(n):
    if n == "A1":
        return group_A1 ,
    elif n == "A2":
        return group_A2,
    elif n == "B1":
        return group_B1,
    elif n == "B2":
        return group_B2,
    elif n == "C1":
        return group_C1,
    elif n == "C2":
        return group_C2,
    else:
        print("اختار المستوي المناسب لك A1, A2, B1, B2, C1, C2")
    return n

#
#
#course recommendation
#
#
A1 = [ 
      " https://freeenglishlessonplans.com/tag/a1/",
      "https://www.elllo.org/index-levels.htm\n https://eslteenstuff.com/lessons-level-elementary-a1-1-a2/",
      "http://www.e4thai.com/e4e/images/pdf/english-out-there-ss1-beginner-level-1-english.pdf",
      "https://languageadvisor.net/english-for-everyone-level-1-beginner-course-book/",
      "https://www.pdfdrive.com/english-unlimited-starter-a1-coursebook-e186433948.html",
      "https://youtube.com/playlist?list=PLRgsws9rC3IWoilPWpOCD_NSZC9huKgJS"]

A2 = [
    
    "https://eslteenstuff.com/esl-teens-lessons-level-pre-intermediate-a2-b1/",
    "https://www.interlangues.ch/wp-content/uploads/2020/10/englishfile_4e_preintermediate_teachers_guide.pdf",
    "https://www.esl-lounge.com/level2.php",
    "https://www.esolcourses.com/content/topicsmenu/pre-intermediate.html",
    "https://learnenglish.britishcouncil.org/vocabulary/a1-a2-vocabulary",
    "https://learnenglish.britishcouncil.org/grammar/a1-a2-grammar",
    "https://www.pdfdrive.com/english-unlimited-elementary-a2-coursebook-with-e-portfolio-e188407766.html",
   "https://www.pdfdrive.com/the-good-grammar-book-a-grammar-pactice-book-for-elementary-to-lower-intermediate-students-of-english-e157043944.html",
   "https://youtube.com/playlist?list=PLRgsws9rC3IVOhtzpsF2IpL0zNZ4X97y-"]

B1 = ["https://learnenglish.britishcouncil.org/english-level/b1-english-level-intermediate",
    "https://www.teachingenglish.org.uk/resources/secondary/lesson-plans/intermediate-b1",
    "https://www.ieltstestonline.com/courses/free-english-course-level-b1/",
    "https://www.pdfdrive.com/english-unlimited-pre-intermediate-b1-students-book-d187308780.html",
    "https://ateneum.edu.pl/assets/Dziekanat/ELEARNING/Kolendo/New-Total-English-Intermediate-Students-Book.pdf",
    "https://youtube.com/playlist?list=PLRgsws9rC3IXIFeqzWF9D0bRRoAkHwj8P"]
B2 = [
    "https://www.pdfdrive.com/english-unlimited-upper-intermediate-b2-students-book-e187260025.html",
    "https://linguapress.com/inter.htm ",
    "https://learnenglish.britishcouncil.org/english-level/b2-english-level-upper-intermediate",
    "https://www.mindluster.com/certificate/855 ",
    "https://perfectlyspoken.com/english-courses/b2/",
    "https://www.ieltstestonline.com/courses/free-english-course-level-b2/",
    "https://www.cambridgeenglish.org/Images/167791-b2-first-handbook.pdf",
    "https://youtube.com/playlist?list=PLRgsws9rC3IWbYVaAvWWbuRugovY08S15",
    "pdfdrive.com/use-of-english-b2-for-all-exams-students-book-e187290797.html"]

C1 = [
    "https://learnenglish.britishcouncil.org/english-level/c1-english-level-advanced",
    "https://www.languages247.com/advanced-english-course-c1-index/",
    "https://freeenglishlessonplans.com/category/advanced-c1/",
    "https://www.youtube.com/watch?v=P2pSnGEcssQ&list=PLRgsws9rC3IW72BgJVfxU2j1cVTgTW2JM"]

C2 = [
    "https://www.pdfdrive.com/advanced-english-phrasal-verbs-in-use-e33416672.html",
    "https://www.pdfdrive.com/english-advanced-vocabulary-and-structure-practice-e16681388.html"]

def course_recommendation(n):
    if n == "A1":
        return  random.choice(A1),
    elif n == "A2":
        return  random.choice(A2),
    elif n == "B1":
        return  random.choice(B1),
    elif n == "B2":
        return  random.choice(B2),
    elif n == "C1":
        return  random.choice(C1),
    elif n == "C2":
        return  random.choice(C2),
    else:
        return "لا يمكنك الاختيار الا من خلال المستويات المتاحة"

def class_type(inp):
    result_class = predict_class(inp,tokenized_words)
    return result_class
def chat(inp):
    while True:
        inp_clean = bag_of_words(inp, tokenized_words)
        res = model.predict(np.array([inp_clean]))[0]
        res_index=np.argmax(res)
        #print(res[res_index])
        if res[res_index]>0.7:
            for tg in data["intents"]:
                #print(result_class)
                if class_type(inp) == "take exam":
                    print("Please choose your level.")
                    return 
                elif class_type(inp) == "peer grouping":
                    return 
                elif class_type(inp) == "course recommendation":
                    return 
                elif class_type(inp) == tg['Intent']:
                    response = tg['purpose']

            return (random.choice(response))
        else:
            return "للأسف مش فاهمك,ممكن تعيد الكلام بطريقه اوضح؟ "
    