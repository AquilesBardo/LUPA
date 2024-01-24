import pandas as pd
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import streamlit as st

# Leer el archivo CSV
df = pd.read_csv("output_selenium_all_data_TextoCompleto.csv")

# Dividir la columna 'Subjects' por punto y coma y expandir en columnas
Subjects_unique_2 = df['Subjects'].str.split(';', expand=True)

# Crear un DataFrame vacío para almacenar los elementos únicos
elementos_unicos_2 = pd.DataFrame()

# Iterar sobre las columnas y concatenar los valores en el DataFrame
for col in range(Subjects_unique_2.shape[1]):
    elementos_unicos_2 = pd.concat([elementos_unicos_2, Subjects_unique_2[col]], axis=0)

# Obtener los 7 elementos con mayor número de ocurrencias
top_7_elementos = elementos_unicos_2.value_counts().nlargest(7).index.tolist()

# Limpiar los elementos seleccionados
top_7_elementos_cleaned = [str(element)[1:-1].replace("'", "").replace(",", "").strip() for element in top_7_elementos]

# Interfaz de Streamlit
st.title('Filtrar DataFrame por Elemento')

# Botones para seleccionar el elemento
selected_element = st.radio('Seleccione un elemento:', top_7_elementos_cleaned)

# Filtrar el DataFrame según el elemento seleccionado en la columna "Subjects"
df_filtrado = df[df['Subjects'].apply(lambda x: selected_element in map(str.strip, str(x).split(';')))]

# Mostrar el DataFrame filtrado
st.write("\nDataFrame filtrado para el elemento '{}':".format(selected_element))
st.write(df_filtrado)

# Cargar modelo Word2Vec y funciones de limpieza
# Asegúrate de que estas líneas estén en el mismo script o archivo
df_filtrado['Texto Completo'].fillna('', inplace=True)

english_stopwords = set(stopwords.words('english'))

def clean_text(text):
    words = word_tokenize(str(text).lower())
    filtered_words = [word for word in words if word.isalpha() and word not in english_stopwords and word not in ['x', 'y', 'et', 'al', 'p']]
    return filtered_words

# Aplicar la limpieza a la columna 'Texto Completo'

df_filtrado['Cleaned Text'] = df_filtrado['Texto Completo'].apply(clean_text)

# Modelo 
model = Word2Vec(df_filtrado['Cleaned Text'], vector_size=100, window=5, min_count=1, workers=4)

# Interfaz de Streamlit
st.title('Filtrar DataFrame por Elemento')

# Buscador
search_query = st.text_input('Buscar por palabra clave:')
search_query_cleaned = clean_text(search_query)

# Limpiar el input del usuario
cleaned_user_input = search_query_cleaned 

# Obtener el embedding del input del usuario
user_input_embedding = np.mean([model.wv[word] for word in cleaned_user_input if word in model.wv], axis=0)

# Eliminar filas con NaN en la columna "Cleaned Text"
df_dropna = df_filtrado.dropna(subset=['Cleaned Text'])

# Definición de la función calculate_similarity
def calculate_similarity(x, model, user_input_embedding):
    word_vectors = [model.wv[word] for word in x if word in model.wv]

    if not word_vectors:
        return np.nan  # o podrías devolver un valor por defecto para representar "sin similitud"

    # Calcular la media solo si la lista no está vacía
    mean_vector = np.mean(word_vectors, axis=0)

    # Verificar si mean_vector es un array no nulo y no contiene NaN antes de calcular la similitud
    if mean_vector is not None and not np.isnan(mean_vector).any():
        # Reshape a (1, -1) para convertir la matriz 1D en 2D
        mean_vector_2d = mean_vector.reshape(1, -1)
        
        # Asegurarse de que user_input_embedding tenga la misma dimensión que mean_vector
        user_input_embedding_reshaped = user_input_embedding.reshape(1, -1)
        
        # Asegurarse de que ambos vectores tengan la misma dimensión
        if user_input_embedding_reshaped.shape == mean_vector_2d.shape:
            # Calcular la similitud coseno
            similarity = cosine_similarity(user_input_embedding_reshaped, mean_vector_2d)[0][0]
            
            # Manejar el caso en el que la similitud no se pueda calcular correctamente
            if np.isnan(similarity):
                return np.nan
            
            return similarity

    return np.nan

# Aplicar la función calculate_similarity
df_dropna['Similarity'] = df_dropna['Cleaned Text'].apply(lambda x: calculate_similarity(x, model, user_input_embedding))

# Ordenar el DataFrame por similitud en orden descendente
df_sorted = df_dropna.sort_values(by='Similarity', ascending=False)

# Sacar los textos del dataframe
columnas_a_excluir = ["Texto Completo", "Cleaned Text"]
df_sorted_sin_textos = df_sorted.drop(columns=columnas_a_excluir)

# Mostrar el DataFrame ordenado
st.write("Filas del DataFrame ordenadas por similitud al input del usuario:")
st.write(df_sorted_sin_textos)