# LUPA™

LUPA™ es un buscador interactivo orientado a las publicaciones investigativas de  arxiv con la temática de “Language and Computation”.

## Requerimientos
pip install numpy
pip install pandas
pip install streamlit
pip install nltk
pip install gensim
nltk.download("stopwords")
nltk.download("punkt)

## Funciones 
Buscador por palabra
Buscador por tema
Agrupar temas

## Usos en investigación 
ORGANIZACION: Organizar la bibliografía
Filtrar grandes cantidades de datos
DISTANT READING: Salvar tiempo viendo los datos de mayor relevancia de una gran cantidad de datos
Skimming
BUSQUEDA: Ofrece más opciones y es más personalizado con los objetivos 
Permite buscar cruces y subcategorías
Profundizar en la búsqueda
MULTIDISCIPLINARIO: Permite hallar cruces y áreas en común entre diferentes temas 

## Explicación de los elementos
El archivo "output_selenium_all_data_TextoCompleto.csv" fue obtenido por medio del código "scrape_arxiv.py"

El archivo "word2vec_model" se consiguió por medio del código "model.py"

Para correr el código solo hay que guardar "lupa_final.py" y correr el comando "streamlit run lupa_final.py". Asegúrese que se encuentra en el directorio del archivo cuando se ejecute

## Explicación de uso
Espere unos segundos para que el programa termine de correr por completo, luego, haga click en el subject que desea utilizar como criterio para aplicar el filtro. Opcionalmente puede escribir una palabra en el buscador, el programa le mostrará en orden descendente los papers en función de la similitud de su texto completo con el input.