############################################
# User-Based Collaborative Filtering
#############################################

# 1. Preparation of the Dataset
# 2. Determination of the Movies Watched by the random User for whom Recommendation will be Made
# 3. Accessing the Data and IDs of Other Users who Watched the Same Movies
# 4. Determination of Users with the Most Similar Behaviors to the User for whom Recommendation will be Made
# 5. Calculation of the Weighted Average Recommendation Score

#############################################
# 1. Preparation of the Dataset
#############################################

import pandas as pd

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)
# pd.set_option('display.float_format', lambda x: '%.5f' % x)

movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')

df = movie.merge(rating, how='left', on='movieId')

comment_counts = pd.DataFrame(df['title'].value_counts()) # [27262 rows x 1 columns]
comment_counts.columns = ['count']

common_movies = comment_counts[comment_counts['count'] > 1000].index # [3159 rows]
common_movies = df[df['title'].isin(common_movies)] # [17766015 rows x 6 columns]

df.shape # (20000797, 6)
common_movies.shape # (17766015, 6)

user_movie_df = common_movies.pivot_table(index=['userId'], columns=['title'], values='rating')

user_movie_df.head().iloc[:5,:5]

#############################################
# 2. Determination of the Movies Watched by the random User for whom Recommendation will be Made
#############################################

random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)

random_user_df = user_movie_df[user_movie_df.index == random_user]

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
len(movies_watched) # 33

random_user_info = df[df['userId'] == random_user][['title', 'rating']]

random_user_info.reset_index(drop='index').head()

#                                    title  rating
# 0                         Sabrina (1995)     5.0
# 1         American President, The (1995)     3.0
# 2           Sense and Sensibility (1995)     5.0
# 3  Ace Ventura: When Nature Calls (1995)     2.0
# 4                            Babe (1995)     5.0

# user_movie_df.loc[(user_movie_df.index == random_user), (user_movie_df.columns == 'Secret Garden, The (1993)')]

#############################################
# 3. Accessing the Data and IDs of Other Users who Watched the Same Movies
#############################################
user_movie_df.head().iloc[:5,:5]

movies_watched_df = user_movie_df[movies_watched]

user_movie_count = movies_watched_df.T.notnull().sum()

user_movie_count = user_movie_count.reset_index()

user_movie_count.columns = ['userId', 'movie_count']

user_movie_count[user_movie_count['movie_count'] > 20].sort_values('movie_count', ascending=False) # 3202

users_same_movies = user_movie_count[user_movie_count['movie_count'] > 20]['userId']

#############################################
# 4. Determination of Users with the Most Similar Behaviors to the User for whom Recommendation will be Made
#############################################

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)], random_user_df[movies_watched]])

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=['corr'])

corr_df.index.names = ['user_id_1', 'user_id_2']

corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df['user_id_1'] == random_user) & (corr_df['corr'] >= 0.65)][['user_id_2', 'corr']].reset_index(drop=True)

top_users = top_users.sort_values('corr', ascending=False)

top_users.rename(columns={'user_id_2': 'userId'}, inplace=True)

top_users_ratings = top_users.merge(rating[['userId', 'movieId', 'rating']], how='inner')

top_users_ratings = top_users_ratings[top_users_ratings['userId'] != random_user]

#############################################
# 5. Calculation of the Weighted Average Recommendation Score
#############################################

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

top_users_ratings.groupby('movieId').agg({'weighted_rating': 'mean'})

recommendation_df = top_users_ratings.groupby('movieId').agg({'weighted_rating': 'mean'})

recommendation_df = recommendation_df.reset_index()

recommendation_df[recommendation_df['weighted_rating'] > 3.5]

movies_to_be_recommend = recommendation_df[recommendation_df['weighted_rating'] > 3.5].sort_values('weighted_rating', ascending=False)

movies_to_be_recommend.merge(movie[['movieId', 'title']])

#     movieId  weighted_rating                                              title
# 0        30         3.952023  Shanghai Triad (Yao a yao yao dao waipo qiao) ...
# 1       326         3.952023                            To Live (Huozhe) (1994)
# 2       242         3.952023                      Farinelli: il castrato (1994)
# 3        53         3.952023                                    Lamerica (1994)
# 4      1348         3.692678  Nosferatu (Nosferatu, eine Symphonie des Graue...
# 5       501         3.679739                                       Naked (1993)
# 6     25850         3.664714                                     Holiday (1938)
# 7      8128         3.664714                       Au revoir les enfants (1987)
# 8     26394         3.664714                          Turning Point, The (1977)
# 9      7585         3.664714                                  Summertime (1955)
# 10     7490         3.664714  At First Sight (Entre Nous) (Coup de foudre) (... 
