# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
#from gensim.models import Word2Vec

# collection of functions useful for preprocessing                                                               """
import os
import io
import re
import string
import multiprocessing
from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk import download
stops = set(stopwords.words("english"))
stop_words = stopwords.words('english')
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')
download('punkt') 
num_features=300
min_word_counts=1
num_workers=multiprocessing.cpu_count()
context_size=5
#size=200
#workers=4
#downsampling=1e-3
#seed=1
#my_rule=3


def get_data(dirname):
    """ Collects all text files in the given folder   """
    if not os.listdir(dirname):
        print ("Files not found-Empty Directory ")
        return
    else:
        files = os.listdir(dirname)
    filenames = [dirname+"/"+files[i] for i in range(len(files))]
    train_data = [io.open(filenames[i], 'r', encoding='UTF-8').read() for i in range(len(filenames))]
#    print (type(train_data))
    return train_data
##
def remove_characters_after_tokenization(tokens):
    """ takes a list of teokens and removes special characters """
    pattern = re.compile('([{}])*'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    return filtered_tokens

##
def text_to_list(dirname):
    """ reads text files in a directory into a  list """
    filenames = os.listdir(dirname)
    files_list = [open(dirname+"/"+filenames[i], "r",encoding="utf-8").readlines() for i in range(len(filenames))]
    #print (files_list)
    return files_list
#
##
def list_txt(lis,txt = 'output.txt'):
    """ writes a list to text file where each element is in separate line """
    thefile = open(txt,'w',encoding="utf-8")
    for item in lis:
      thefile.write("%s\n" % item)
    thefile.close()
##
###
def preprocess_documents(file_lists):
    """ takes a list of documents as list and preprocess them """
    file_tokens =[[file_lists[i][j].lower().split() for j in range(len(file_lists[i]))] for i in range(len(file_lists))]
#    print (file_tokens)
    file_tokens_clean = [[x for x in file_tokens[i] if x] for i in range(len(file_tokens))]
#    print (file_tokens_clean)
    file_tokens_clean2 = [[remove_characters_after_tokenization(file_tokens_clean[i][j]) for j in range(len(file_tokens_clean[i]))] for i in range (len(file_tokens_clean))]
    files_final = [[[w for w in file_tokens_clean2[i][j]  if not w in stops] for j in range(len(file_tokens_clean2[i]))] for i in range(len(file_tokens_clean2))]
#    files_lemma = [[[wnl.lemmatize(files_final[i][j][k].decode('UTF-8')) for k in range(len(files_final[i][j]))] for j in range(len(files_final[i]))] for i in range(len(files_final))]
    docs = [[[files_final[i][j][k].encode('utf-8','ignore') for k in range(len(files_final[i][j]))] for j in range(len(files_final[i]))] for i in range(len(files_final))]
    docs = [[[files_final[i][j][k] for k in range(len(files_final[i][j]))] for j in range(len(files_final[i]))] for i in range(len(files_final))]
    docs_corpus = [[" ".join(docs[i][j]) for j in range(len(docs[i]))] for i in range(len(docs))] 
    docs_final = [" ".join(docs_corpus[j])for j in range(len(docs_corpus))] 
#    print (type(docs_final))
    return docs_final

def preprocess_query(query):
    """ takes a list of query(single line) as list and preprocess them """
    """ takes a list of query(single line) as list and preprocess them """
    query_tokens = [query[i].lower().split() for i in range(len(query))]
    query_tokens_clean = [[x for x in query_tokens[i] if x] for i in range(len(query_tokens))]
    query_tokens_clean2 = [remove_characters_after_tokenization(query_tokens_clean[j]) for j in range(len(query_tokens_clean))] 
    query_final = [[w for w in query_tokens_clean2[i] if not w in stops] for i in range(len(query_tokens_clean2))]
#    print(query_final)
#    query_lemma = [[wnl.lemmatize(query_final[i][j].decode('latin-1')) for j in range(len(query_final[i]))] for i in range(len(query_final))]
    query = [[query_final[i][j].encode('utf-8','ignore') for j in range(len(query_final[i]))] for i in range(len(query_final))]
    query_corpus = [" ".join(query_final[i]) for i in range(len(query_final))]
#    print (query_corpus)
    return query_corpus
def preprocess_gensim(doc):
    """ preprocess raw text by tokenising and removing stop-words,special-charaters """
    
      # Lower the text.
#    doc = doc.lower  # Lower the text.
    doc = word_tokenize(doc)  # Split into words.
    
    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
    doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.
    print (doc)
    return doc
###
##
def train_word2vec(train_data,worker_no=3,min_count=min_word_counts, vector_size=num_features,model_name="word2vec_model"):
    """ Trains a word2vec model on the preprocessed data and saves it . """
     
    
    if not train_data:
        print ("no training data")
        return
    w2v_corpus = [preprocess_gensim(train_data[i]) for i in range(len(train_data))]
    
    model = Word2Vec(size=200,window=5,min_count=1,workers=5,alpha=0.025,sg=0,hs=0,negative=5,cbow_mean=1,iter=20)
    print ("Model Created Successfully")
    model.build_vocab(w2v_corpus)
    print ("word2vec vocuablary length", len(model.wv.vocab))
    
   
#    model.train(w2v_corpus,total_examples=model.corpus_count,epochs=model.iter)
#    print ("model trained")
# 
    if not os.path.exists("trained"):
        os.mkdir("trained")
#    
    model.save(os.path.join("trained","model.train"))
    print ("saving the files to the path")
    model.wv.save_word2vec_format(r"C:\Users\user\trained\word2vec_model.train.bin",binary=True)
    print ("model saved in the bin format")
    model.wv.save_word2vec_format(r"C:\Users\user\trained\word2vec_model.traina.txt",binary=False)
    print ("model saved in the txt fomat")




if __name__ == '__main__':
    dirname = r"F:\word\we"
    train_data=get_data(dirname)
    print (train_data)
    Ttolist=text_to_list(dirname)
    print (Ttolist)
    rtofile=list_txt(Ttolist)
    print (rtofile)
##    listtofile=r'C:\Users\user\Desktop\mypre\output'
    poutput=preprocess_documents(Ttolist)
    print (poutput)
    qresult=str(preprocess_query( poutput))
    print (qresult)
    doc1=preprocess_gensim(qresult)
    print (doc1)
    train_word2vec(doc1)
   
    
    



