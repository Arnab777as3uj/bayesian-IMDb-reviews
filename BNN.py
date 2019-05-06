
# coding: utf-8

# In[128]:
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import pandas as pd
import numpy as np
import os
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('tagsets')
nltk.download('wordnet')
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

rev_all = pd.read_csv('../input/imdb_master.csv',encoding="ISO-8859-1",index_col=None)
del rev_all['Unnamed: 0']

rev_all_unsup = rev_all[rev_all['label'] == 'unsup']

rev_all = rev_all[~(rev_all['label'] == 'unsup')]

rev_all.review = rev_all.review.str.lower()
rev_all.review = rev_all.review.str.replace(r'<br />',' ')
rev_all = rev_all[~rev_all.review.str.match(r'^\s*$')]
rev_all.review = rev_all.review.str.replace('[^a-zA-Z]',' ')
rev_all.review = rev_all.review.str.replace(r'\n',' ')
rev_all.review = rev_all.review.str.replace(r'\s+',' ')

filtered_reviews = []
#stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

for review in rev_all.review:

     filtered_reviews.append([ps.stem(w.lower()) for w in word_tokenize(review)])

corpus = [' '.join(x) for x in filtered_reviews]

vect = TfidfVectorizer(min_df=0.0005,lowercase=True,stop_words='english',norm='l1',ngram_range=(1,2),max_features=1500).fit(corpus)

X = vect.transform(corpus)

X_df = pd.DataFrame(X.toarray())

X_df


X_train, X_test, y_train, y_test = train_test_split(X_df, 
                                                    rev_all['label'], 
                                                    random_state=7, test_size=0.25)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=1)  
X_train = lda.fit_transform(X_train, y_train)  
X_test = lda.transform(X_test)  

clf = LDA()
clf.fit(X_train, y_train)
LDA(n_components=None, priors=None, shrinkage=None,    
    solver='svd', store_covariance=False, tol=0.0001)
pred = clf.predict(X_test)

np.unique(pred,return_counts=True)

acc = sklearn.metrics.accuracy_score(np.array(y_test), 
                                     np.array(pred))

print(np.mean(pred==y_test))

acc

confusion_matrix(y_test, pred)

sns.heatmap(pd.DataFrame(confusion_matrix(y_test, pred)), annot=True)

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np

q_clf = QuadraticDiscriminantAnalysis()
q_clf.fit(X_train, y_train)
QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                              store_covariance=False,
                              store_covariances=None, tol=0.0001)
q_pred = q_clf.predict(X_test)

np.unique(q_pred,return_counts=True)

print(np.mean(q_pred==y_test))

sns.heatmap(pd.DataFrame(confusion_matrix(y_test, q_pred)), annot=True)

from keras import backend as K
from keras import activations, initializers
from keras.layers import Layer

import tensorflow as tf

def mixture_prior_params(sigma_1, sigma_2, pi, return_sigma=False):
    params = K.variable([sigma_1, sigma_2, pi], name='mixture_prior_params')
    sigma = np.sqrt(pi * sigma_1 ** 2 + (1 - pi) * sigma_2 ** 2)
    return params, sigma

def log_mixture_prior_prob(w):
    comp_1_dist = tf.distributions.Normal(0.0, prior_params[0])
    comp_2_dist = tf.distributions.Normal(0.0, prior_params[1])
    comp_1_weight = prior_params[2]    
    return K.log(comp_1_weight * comp_1_dist.prob(w) + (1 - comp_1_weight) * comp_2_dist.prob(w))    

# Mixture prior parameters shared across DenseVariational layer instances
prior_params, prior_sigma = mixture_prior_params(sigma_1=1.0, sigma_2=0.1, pi=0.2)

class DenseVariational(Layer):
    def __init__(self, output_dim, kl_loss_weight, activation=None, **kwargs):
        self.output_dim = output_dim
        self.kl_loss_weight = kl_loss_weight
        self.activation = activations.get(activation)
        super().__init__(**kwargs)

    def build(self, input_shape):  
        self._trainable_weights.append(prior_params) 

        self.kernel_mu = self.add_weight(name='kernel_mu', 
                                         shape=(input_shape[1], self.output_dim),
                                         initializer=initializers.normal(stddev=prior_sigma),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu', 
                                       shape=(self.output_dim,),
                                       initializer=initializers.normal(stddev=prior_sigma),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho', 
                                          shape=(input_shape[1], self.output_dim),
                                          initializer=initializers.constant(0.0),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho', 
                                        shape=(self.output_dim,),
                                        initializer=initializers.constant(0.0),
                                        trainable=True)
        super().build(input_shape)

    def call(self, x):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)
                
        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) + 
                      self.kl_loss(bias, self.bias_mu, bias_sigma))
        
        return self.activation(K.dot(x, kernel) + bias)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def kl_loss(self, w, mu, sigma):
        variational_dist = tf.distributions.Normal(mu, sigma)
        return kl_loss_weight * K.sum(variational_dist.log_prob(w) - log_mixture_prior_prob(w))



from keras.layers import Input
from keras.models import Model
train_size = 37500
batch_size = train_size
num_batches = train_size / batch_size
kl_loss_weight = 1.0 / num_batches

x_in = Input(shape=(1,))
x = DenseVariational(20, kl_loss_weight=kl_loss_weight, activation='relu')(x_in)
x = DenseVariational(20, kl_loss_weight=kl_loss_weight, activation='relu')(x)
x = DenseVariational(1, kl_loss_weight=kl_loss_weight, activation='sigmoid')(x)

model = Model(x_in, x)


# In[202]:


y_train = y_train.replace({'pos':0,'neg':1})
y_train = y_train.astype('category')
y_train
# In[ ]:

import keras
from keras import callbacks, optimizers
noise = 1.0
def neg_log_likelihood(y_true, y_pred, sigma=noise):
    dist = tf.distributions.Normal(loc=y_pred, scale=sigma)
    return K.sum(-dist.log_prob(y_true))

model.compile(loss=keras.losses.binary_crossentropy, optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, epochs=1000, verbose=0)

import tqdm
y_pred_list = []

for i in tqdm.tqdm(range(500)):
    y_pred = model.predict(X_test)
    y_pred_list.append(y_pred)
    
y_pred
y_test_num = y_test.replace({'pos':0,'neg':1})
print(np.mean(np.round(y_pred).reshape(12500)==y_test_num))

sns.heatmap(pd.DataFrame(confusion_matrix(y_test_num, np.round(y_pred).reshape(12500))), annot=True)