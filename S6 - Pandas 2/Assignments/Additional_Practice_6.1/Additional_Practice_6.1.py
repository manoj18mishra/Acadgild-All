#### Get Data from the following link: http://files.grouplens.org/datasets/movielens/ml-20m.zip
#### We will be using the following files for this exercise:
    #ratings.csv : userId,movieId,rating, timestamp
    #tags.csv : userId,movieId, tag, timestamp
    #movies.csv : movieId, title, genres
#### 1. Read the dataset using pandas.
import pandas as pd
import os

path="C:\\Users\\manoj\\Documents\\Acadgild DSB\\S6 - Pandas 2\\Assignments\\Additional_Practice_6.1"
os.chdir(path)

ratings=pd.read_csv("ratings.csv")
tags=pd.read_csv("tags.csv")
movies=pd.read_csv("movies.csv")

#### 2. Extract the first row from tags and print its type.
tagsFirstRow=tags.iloc[0]
print(type(tagsFirstRow))
#### 3. Extract row 0, 11, 2000 from tags DataFrame.
tags.iloc[[0,11,2000]]
#### 4. Print index, columns of the DataFrame.
print(ratings.index.values)
print(ratings.columns.values)
print(tags.index.values)
print(tags.columns.values)
print(movies.index.values)
print(movies.columns.values)
#### 5. Calculate descriptive statistics for the 'ratings' column of the ratings DataFrame. Verify using describe().
print("count\t"+str(ratings.rating.count()))
print("mean\t"+str(ratings.rating.mean()))
print("std\t"+str(ratings.rating.std()))
print("min\t"+str(ratings.rating.min()))
print("25%\t"+str(ratings.rating.quantile(0.25)))
print("50%\t"+str(ratings.rating.quantile(0.5)))
print("75%\t"+str(ratings.rating.quantile(0.75)))
print("max\t"+str(ratings.rating.max()))
print(ratings.rating.describe())
#### 6. Filter out ratings with rating > 4
ratings[ratings.rating>4]
#### 7. Find how many null values, missing values are present. Deal with them. Print out how many rows have been modified.
    #Checking each dataframe
print("ratings has null:- "+str(ratings.isnull().any().any()))
print("tags has null:- " +str(tags.isnull().any().any()))
print("movies has null:- " +str(movies.isnull().any().any()))
    #Removing null rows
oldCount=tags.movieId.count()
noOfNullRows = tags[tags.tag.isnull()].movieId.count()
tags.dropna(inplace=True)
newCount=tags.movieId.count()
    #Check if the correct count has been removed
print("Old Count:- " + str(oldCount))
print("New Count:- " + str(newCount))
print("Null Count:- " + str(noOfNullRows))
print("Number of Null count matches with old count - new count? :- " + str(noOfNullRows ==(oldCount-newCount)))
#### 8. Filter out movies from the movies DataFrame that are of type 'Animation'.
movies[movies['genres']=="Animation"]
#### 9. Find the average rating of movies.
import numpy as np
print(ratings.groupby("movieId")["rating"].agg(np.mean))
#### 10. Perform an inner join of movies and tags based on movieId.
result=pd.merge(movies,tags,on="movieId")
result
#### 11. Print out the 5 movies that belong to the Comedy genre and have rating greater than 4.
movies_ratings=pd.merge(movies,ratings,on="movieId")
comedy_4=movies_ratings[(movies_ratings["genres"].str.contains("Comedy")) & (movies_ratings["rating"]>4)].groupby("movieId")
comedy_4.head(5)
#### 12. Split 'genres' into multiple columns.
m=pd.concat([movies.drop(["genres"],axis=1),pd.DataFrame(movies["genres"].str.split("|").tolist()).apply(pd.Series).rename(columns={0:"genre_0",1:"genre_1",2:"genre_2",3:"genre_3",4:"genre_4",5:"genre_5",6:"genre_6",7:"genre_7",8:"genre_8",9:"genre_9"})],axis=1)
m.head()
#### 13. Extract year from title e.g. (1995).
y1 = movies['title'].str.extract('(\(\d{4}\))',expand=True)
y2 = movies['title'].str.extract('(\d{4})',expand=True)
#### 14. Select rows based on timestamps later than 2015-02-01.
t=pd.concat([tags.drop(["timestamp"],axis=1),pd.to_datetime(tags['timestamp'], unit='s')],axis=1)
t1=t[t['timestamp']>'2015-02-01']
#### 15. Sort the tags DataFrame based on timestamp.
tags.sort_values("timestamp")
