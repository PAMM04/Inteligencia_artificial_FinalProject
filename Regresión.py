import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Función para cargar datos desde un archivo CSV
def cargar_datos(ruta_csv):
    return pd.read_csv(ruta_csv)

# Función para entrenar el modelo de regresión lineal
def entrenar_modelo(datos, columnas_caracteristicas, columna_objetivo):
    X = datos[columnas_caracteristicas]  # Características (variables predictoras)
    y = datos[columna_objetivo]  # Columna objetivo (precio)
    
    # Se divide en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizamos los datos para que todas las características tengan la misma escala
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Creamos el modelo de regresión lineal
    modelo = LinearRegression()
    modelo.fit(X_train_scaled, y_train)  # Entrenamos el modelo
    
    # Realizamos las predicciones sobre el conjunto de prueba
    y_pred = modelo.predict(X_test_scaled)
    
    # Calculamos el error cuadrático medio
    error = mean_squared_error(y_test, y_pred)
    print(f'Error cuadrático medio: {error}')
    
    # Calculamos el R^2 (coeficiente de determinación)
    r2 = modelo.score(X_test_scaled, y_test)
    print(f'R^2: {r2}')
    
    # Gráfico de Predicciones vs. Valores Reales
    plt.scatter(y_test, y_pred)
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title('Predicciones vs. Valores Reales')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Línea y=x
    plt.show()
    
    return modelo, scaler

# Función para realizar predicciones con el modelo entrenado
def predecir(modelo, scaler, nuevos_datos, columnas_entrenamiento):
    # Verifica si las columnas categóricas están presentes en los nuevos datos
    columnas_categoricas = ['Brand', 'Processor', 'GPU', 'Operating System', 'Resolution']
    
    # Asegúrate de que las columnas categóricas estén en los nuevos datos
    for col in columnas_categoricas:
        if col not in nuevos_datos.columns:
            nuevos_datos[col] = ''  # Añadir columnas faltantes con valores vacíos

    # Aplicamos One-Hot Encoding a los nuevos datos de la misma forma que en el entrenamiento
    nuevos_datos_encoded = pd.get_dummies(nuevos_datos, columns=columnas_categoricas, drop_first=True)
    
    # Aseguramos que las columnas de los nuevos datos coincidan con las columnas del entrenamiento
    nuevos_datos_encoded = nuevos_datos_encoded.reindex(columns=columnas_entrenamiento, fill_value=0)
    
    # Escalamos los nuevos datos antes de hacer la predicción
    nuevos_datos_escalados = scaler.transform(nuevos_datos_encoded)
    
    # Realizamos la predicción
    return modelo.predict(nuevos_datos_escalados)

# Función para limpiar y convertir la columna 'Storage' a un valor numérico en GB
def limpiar_storage(x):
    if isinstance(x, str):
        # Extrae solo los números del valor de almacenamiento
        x = ''.join(filter(str.isdigit, x))
        return int(x) if x else 0
    return x

# Ejemplo de uso
if __name__ == "__main__":
    ruta_csv = 'laptop_prices.csv'  # Reemplaza con la ruta a tu archivo CSV
    datos = cargar_datos(ruta_csv)
    
    # Limpiamos la columna 'Storage' para convertirla en un valor numérico
    datos['Storage'] = datos['Storage'].apply(limpiar_storage)
    
    # Eliminamos filas con valores nulos en las columnas que estamos utilizando
    datos = datos.dropna(subset=['RAM (GB)', 'Storage', 'Screen Size (inch)', 'Battery Life (hours)', 'Price ($)', 'Brand', 'Processor', 'GPU', 'Resolution', 'Weight (kg)'])
    
    # Convertimos las variables categóricas en variables dummy (One-Hot Encoding)
    datos = pd.get_dummies(datos, columns=['Brand', 'Processor', 'GPU', 'Operating System', 'Resolution'], drop_first=True)
    
    # Columnas de características
    columnas_caracteristicas = ['RAM (GB)', 'Storage', 'Screen Size (inch)', 'Battery Life (hours)', 'Weight (kg)'] + [col for col in datos.columns if col.startswith(('Brand', 'Processor', 'GPU', 'Operating System', 'Resolution'))]
    columna_objetivo = 'Price ($)'  # Columna que contiene el precio
    
    # Entrenamos el modelo
    modelo, scaler = entrenar_modelo(datos, columnas_caracteristicas, columna_objetivo)
    
    # Datos nuevos para predecir (cambia estos valores por los datos que quieras predecir)
    nuevos_datos = pd.DataFrame({
        'RAM (GB)': [64],  # Ejemplo de RAM
        'Storage': [512],  # Ejemplo de almacenamiento (en GB)
        'Screen Size (inch)': [17.3],  # Ejemplo de tamaño de pantalla
        'Battery Life (hours)': [8.9],  # Ejemplo de duración de la batería
        'Weight (kg)': [1.42],  # Ejemplo de peso
        # Asegúrate de incluir las columnas One-Hot codificadas de las variables categóricas
        'Brand_Apple': [1],  # Si es una laptop Apple (ajusta según corresponda)
        'Processor_Intel': [1],  # Si tiene un procesador Intel (ajusta según corresponda)
        'GPU_NVIDIA': [1],  # Si tiene GPU NVIDIA (ajusta según corresponda)
        'Resolution_2560x1440': [1],  # Si tiene resolución 1920x1080 (ajusta según corresponda)
    })
    
    # Realizamos la predicción
    predicciones = predecir(modelo, scaler, nuevos_datos, columnas_caracteristicas)
    print(f'Predicción de precio: {predicciones[0]}')
