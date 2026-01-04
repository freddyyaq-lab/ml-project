import os
import pandas as pd


class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def validate_file(self) -> None:

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"No se encontro el archivo: {self.file_path}")

        if not self.file_path.lower().endswith((".csv", ".xlsx")):
            raise ValueError("El archivo debe ser .csv o .xlsx")

    def load_data(self) -> pd.DataFrame:
        self.validate_file()

        if self.file_path.lower().endswith(".csv"):
            df = pd.read_csv(self.file_path)
        else:
            df = pd.read_excel(self.file_path)

        return df


def cargar_datos(file_path: str) -> pd.DataFrame:
    loader = DataLoader(file_path)
    return loader.load_data()
