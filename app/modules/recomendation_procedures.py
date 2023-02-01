import pandas as pd
import numpy as np
from apyori import apriori
from sklearn.metrics.pairwise import cosine_similarity
import operator
from scipy.sparse.linalg import svds
import pickle


# Función para encontrar k usuarios similares al usuario objetivo 
def similar_users(user_id, matrix, k=5):
    # Crea una matriz con el usuario a evaluar
    user = matrix[matrix.index == user_id]
    
    # Crea una matriz con los otros usuarios
    other_users = matrix[matrix.index != user_id]
    
    # Calcula la similaridad de coseno entre el usuario a evaluar y los demás usuarios
    similarities = cosine_similarity(user,other_users)[0].tolist()
    
    # Crea una lista de indices de estos usuarios
    indices = other_users.index.tolist()
    
    # Crea un diccionario de pares entre indices de usuarios y sus similaridades
    index_similarity = dict(zip(indices, similarities))
    
    # Ordena por similaridad
    index_similarity_sorted = sorted(index_similarity.items(), key=operator.itemgetter(1))
    index_similarity_sorted.reverse()
    
    # Selecciona el top k de usuarios
    top_users_similarities = index_similarity_sorted[:k]
    users = [u[0] for u in top_users_similarities]
    
    return users

# Funcion para recomendar los productos a aprtir de usuarios similares
def recommend_item(user_index, similar_user_indices, matrix, items=5):

    # Selecciona los vectores de usuarios similares
    similar_users = matrix[matrix.index.isin(similar_user_indices)]
    # Calcula la media sobre los usuarios similares
    similar_users = similar_users.mean(axis=0)
    # Convierte a DataFrame para facilitar ordenado y filtro
    similar_users_df = pd.DataFrame(similar_users, columns=['mean'])
    
    # Selecciona el vector del usuario actual
    user_df = matrix[matrix.index == user_index]
    # Transpone para facilitar el filtrado
    user_df_transposed = user_df.transpose()
    # Renombra la columna como review
    user_df_transposed.columns = ['reviews']
    # Selecciona los prodcutos no comprados hasta el momento
    user_df_transposed = user_df_transposed[user_df_transposed['reviews']==0]
    # Ordena el DataFrame
    similar_users_df_ordered = similar_users_df.sort_values(by=['mean'], ascending=False)
    # Selecciona el top n de indices de productos   
    top_n_productos = similar_users_df_ordered.head(items)
    top_n_productos_indices = top_n_productos.index.tolist()

    return [top_n_productos_indices[0],top_n_productos_indices[1],top_n_productos_indices[2],top_n_productos_indices[3],top_n_productos_indices[4]]


def recomendation(user:str):
    with open('data/recomendation/usuarios_productos_matrix.pickle', 'rb') as file:
        matrix = pickle.load(file)
    return recommend_item(  user, 
                            similar_users(user,matrix), 
                            matrix)