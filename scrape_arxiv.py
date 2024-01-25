from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import pandas as pd

url = "https://arxiv.org/list/cs.CL/pastweek?show=364"

# Configurar el navegador (asegúrate de tener el controlador adecuado instalado)
driver = webdriver.Chrome()

# Acceder a la página
driver.get(url)

# Esperar a que la página cargue completamente (puedes ajustar el tiempo según tu conexión)
driver.implicitly_wait(10)

# Encontrar todos los elementos con la clase "list-title mathjax"
title_elements = driver.find_elements(By.CLASS_NAME, "list-title.mathjax")

# Encontrar todos los elementos con la clase "list-authors"
author_elements = driver.find_elements(By.CLASS_NAME, "list-authors")

# Encontrar todos los elementos con la clase "list-subjects"
subject_elements = driver.find_elements(By.CLASS_NAME, "list-subjects")

# Encontrar todos los elementos con la clase "list-identifier"
identifier_elements = driver.find_elements(By.CLASS_NAME, "list-identifier")

# Crear una lista para almacenar los datos
data = []

# Verificar si se encontraron elementos para títulos, autores, subjects e identifiers
if title_elements and author_elements and subject_elements and identifier_elements:
    # Iterar sobre los elementos y escribir en la lista de datos
    for i in range(len(title_elements)):
        try:
            # Obtener el texto del elemento de título
            title = title_elements[i].text.strip()

            # Obtener el texto del elemento de autores
            authors = author_elements[i].text.strip().replace('Authors:', '')

            # Obtener el texto del elemento de subjects
            subjects = subject_elements[i].text.strip().replace('Subjects:', '')

            # Obtener el enlace del identificador y acceder a él
            identifier_link = identifier_elements[i].find_element(By.TAG_NAME, 'a').get_attribute('href')
            driver.get(identifier_link)

            # Esperar a que la nueva página cargue completamente
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "abstract.mathjax")))

            # Encontrar el elemento con la clase "abstract mathjax" en la nueva página
            abstract_element = driver.find_element(By.CLASS_NAME, "abstract.mathjax")

            # Obtener el elemento con la clase "dateline" en la nueva página
            dateline_element = driver.find_element(By.CLASS_NAME, "dateline")

            # Obtener el texto del elemento de resumen (Abstract) y de la fecha (dateline)
            abstract = abstract_element.text.strip()
            date = dateline_element.text.strip()

            try:
                # Obtener el enlace para descargar el HTML experimental
                html_download_link = driver.find_element(By.ID, "latexml-download-link").get_attribute('href')

                # Acceder a la página del enlace HTML experimental
                driver.get(html_download_link)

                # Esperar a que la nueva página cargue completamente
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "article")))

                # Obtener el texto completo del elemento <article>
                full_text_element = driver.find_element(By.TAG_NAME, "article")
                full_text = full_text_element.text.strip()
            except Exception as e:
                print(f"Error al obtener el texto completo: {str(e)}")
                full_text = "NaN"

            # Agregar los datos a la lista
            data.append([title, authors, subjects, abstract, date, identifier_link, full_text])

            # Volver a la página madre inicial
            driver.get(url)

            # Esperar a que la página madre inicial cargue completamente
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "list-title.mathjax")))
            
            # Encontrar nuevamente los elementos después de volver a la página madre inicial
            title_elements = driver.find_elements(By.CLASS_NAME, "list-title.mathjax")
            author_elements = driver.find_elements(By.CLASS_NAME, "list-authors")
            subject_elements = driver.find_elements(By.CLASS_NAME, "list-subjects")
            identifier_elements = driver.find_elements(By.CLASS_NAME, "list-identifier")

        except Exception as e:
            print(f"Error procesando el elemento {i + 1}: {str(e)}")
            # Agregar una fila con NaN en caso de error
            data.append(["NaN"] * 7)

# Crear un DataFrame con los datos
df = pd.DataFrame(data, columns=['Titulo', 'Autores', 'Subjects', 'Resumen', 'Fecha', 'Enlace', 'Texto Completo'])

# Guardar el DataFrame como un archivo CSV
df.to_csv('output_selenium_all_data_updated.csv', index=False)

print("Datos exportados exitosamente a output_selenium_all_data_updated.csv")

# Cerrar el navegador
driver.quit()



