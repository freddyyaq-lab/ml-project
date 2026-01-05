import pandas as pd
import os
import sys

class Cargar_datos:
    """
    Clase encargada de la ingesta de datos desde archivos Excel o CSV.
    """
    
    def __init__(self, file_path: str):
        """
        Inicializa la clase con la ruta del archivo.
        :param file_path: Ruta relativa o absoluta al archivo de datos.
        """
        self.file_path = file_path

    def carga_datos(self) -> pd.DataFrame:
        """
        Carga los datos y devuelve un DataFrame de Pandas.
        Detecta automáticamente si es Excel (.xlsx) o CSV.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Error: El archivo no se encuentra en la ruta: {self.file_path}")

        print(f"Cargando datos desde: {self.file_path}...")

        try:
            if self.file_path.endswith('.xlsx') or self.file_path.endswith('.xls'):
                df = pd.read_excel(self.file_path, engine='openpyxl')
            elif self.file_path.endswith('.csv'):
                df = pd.read_csv(self.file_path)
            else:
                raise ValueError("Formato no soportado. Usa .xlsx o .csv")

            print(f"Carga exitosa. Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas.")
            return df
        
        except Exception as e:
            print(f"Error crítico al cargar datos: {e}")
            sys.exit(1)
