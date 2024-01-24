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
