#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import ast


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head(1)


# In[4]:


credits.head(1)


# In[5]:


movies=movies.merge(credits,on='title')


# In[6]:


#genres
#id
#keywords
#title
#overview
#cast
#crew
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[7]:


movies.head()


# In[8]:


movies.isnull().sum()


# In[9]:


movies.dropna(inplace=True)


# In[10]:


movies.duplicated().sum()


# 

# In[11]:


movies.iloc[0].genres


# In[12]:


#[action,adventure,fantacy,scifi]


# In[13]:


def convert(obj):
    L =[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
    


# In[14]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# 

# In[15]:


movies['keywords']=movies['keywords'].apply(convert)


# In[16]:


movies.head()


# In[17]:


movies['cast'][0]


# In[18]:


def convert3(obj):
    L =[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# 

# In[19]:


movies['cast'] = movies['cast'].apply(convert3)


# 

# In[20]:


movies.head()


# In[21]:


movies['crew'][0]


# In[22]:


def fetch_director(obj):
    L =[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[23]:


movies['crew'] = movies['crew'].apply(fetch_director)


# 

# In[24]:


movies.head()


# In[25]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[26]:


movies.head()


# In[27]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])


# In[28]:


movies.head()


# In[29]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['crew'] + movies['cast']


# In[30]:


movies.head()


# In[32]:


pip install nltk


# In[33]:


new_df = movies[['movie_id','title','tags']]


# In[34]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[35]:


new_df.head()


# In[36]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[37]:


new_df.head()


# In[38]:


import nltk


# In[39]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[40]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[41]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[42]:


new_df['tags'][0]


# In[43]:


new_df['tags'][1]


# In[44]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000,stop_words='english')


# In[45]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[46]:


vectors


# In[47]:


vectors[0]


# In[48]:


cv.get_feature_names_out()


# In[49]:


from sklearn.metrics.pairwise import cosine_similarity


# In[50]:


similarity = cosine_similarity(vectors)


# In[51]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[52]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
       


# In[53]:


recommend('Batman Begins')


# In[54]:


new_df.iloc[1216].title


# In[55]:


import pickle


# In[56]:


pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))


# In[131]:





# In[57]:


pickle.dump(similarity,open('similarity1.pkl','wb'))


# In[ ]:




