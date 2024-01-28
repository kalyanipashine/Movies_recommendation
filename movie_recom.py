import pandas as pd 
import numpy as np 

movies=pd.read_csv("movies.csv")
print(movies.head())
print(movies.describe())
print(movies.info())
print(movies.isnull().sum())
print(movies.columns)

movies=movies[['id', 'title', 'overview', 'genre']]
print(movies)

movies['tags']=movies['overview']+movies['genre']
print(movies)

new_data= movies.drop(columns=['overview','genre'])
print(new_data)

from sklearn.feature_extraction.text import CountVectorizer

cv= CountVectorizer(max_features=10000, stop_words='english')
print(cv)

vector= cv.fit_transform(new_data['tags'].values.astype('U')).toarray()
print(vector.shape)

from sklearn.metrics.pairwise import cosine_similarity

similarity= cosine_similarity(vector)
print(similarity)
#diagonal 1 number means similar data

new_data[new_data['title']== "The Godfather"].index[0]

distance= sorted(list(enumerate(similarity[2])), reverse=True, key=lambda vector:vector[1])
for i in distance[0:5]:
    print(new_data.iloc[i[0]].title)


def recommend(movies):
    index=new_data[new_data['title']== movies].index[0]
    distance= sorted(list(enumerate(similarity[index])), reverse=True,key=lambda vector:vector[1])
    for i in distance[0:5]:
        print(new_data.iloc[i[0]].title)

recommend("Iron Man")

import pickle
pickle.dump(new_data, open('movies_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
pickle.load(open('movies_list.pkl', 'rb'))
