import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class FeatureEngineering:
    """
    Clase encargada de la transformación de datos, limpieza de leakage
    y preparación para el entrenamiento.
    """

    def __init__(self, df: pd.DataFrame, target_col: str = 'Pago_atiempo'):
        """
        :param df: DataFrame con los datos crudos.
        :param target_col: Nombre de la columna objetivo.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None

    def _drop_manual_columns(self):
        """
        Elimina columnas que sabemos por conocimiento de negocio que no sirven
        o son ruidosas.
        AQUÍ SE ELIMINA LA FUGA DE INFORMACIÓN (DATOS DEL FUTURO).
        """
        cols_to_drop = [
            'fecha_prestamo',
            'saldo_mora',
            'saldo_mora_codeudor',
            'saldo_total',
            'saldo_principal',

        ]
        
        existing_cols = [c for c in cols_to_drop if c in self.df.columns]
        if existing_cols:
            print(f"Eliminando columnas manuales: {existing_cols}")
            self.df.drop(columns=existing_cols, inplace=True)

    def _remove_highly_correlated_features(self, threshold=0.90):
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if self.target_col not in numeric_df.columns:
            return 
        
        correlations = numeric_df.corrwith(numeric_df[self.target_col]).abs()
        high_corr_cols = correlations[correlations > threshold].index.tolist()
        
        if self.target_col in high_corr_cols:
            high_corr_cols.remove(self.target_col)
            
        if high_corr_cols:
            print(f"Leakage detectado (> {threshold*100}%). Eliminando: {high_corr_cols}")
            self.df.drop(columns=high_corr_cols, inplace=True)

    def create_pipeline(self):
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_features:
            X[col] = X[col].astype(str)

        print(f"Features Numéricos: {len(numeric_features)}")
        print(f"Features Categóricos: {len(categorical_features)}")
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Desconocido')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Unir en ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        return X, y

    def run_split(self, test_size=0.2, random_state=42):
        self._drop_manual_columns()
        self._remove_highly_correlated_features(threshold=0.85) 
        X, y = self.create_pipeline()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test, self.preprocessor

if __name__ == "__main__":
    try:
        from Cargar_datos import Cargar_datos
    except ImportError:
        try:
            from Cargar_datos import Cargar_datos
        except:
            pass

    # Prueba rápida
    try:
        # Ajusta la ruta a tu Excel
        loader = Cargar_datos("./Base_de_datos.xlsx")
        df = loader.carga_datos()
        fe = FeatureEngineering(df)
        fe.run_split()
        print("Prueba de Feature Engineering exitosa")
    except Exception as e:
        print(f"Error en la prueba: {e}")