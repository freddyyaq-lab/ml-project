import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score
import xgboost as xgb

try:
    from Cargar_datos import Cargar_datos
    from ft_engineering import FeatureEngineering
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from src.Cargar_datos import Cargar_datos
    from src.ft_engineering import FeatureEngineering

class ModelTrainer:
    """
    Clase orquestadora para entrenar.
    AHORA OPTIMIZADA PARA MAXIMIZAR EL RECALL (DETECTAR IMPAGOS).
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.best_model = None
        self.best_model_name = ""
        self.best_metric = 0.0
        self.results = {}

    def run_training(self):
        print("\n" + "="*60)
        print("🚀 INICIANDO ENTRENAMIENTO (ENFOQUE: RECALL / RIESGO)")
        print("="*60)

        loader = Cargar_datos(self.data_path)
        df = loader.carga_datos()

        if df is None: return

        print("\n[2/4] Ingeniería de Features...")
        fe = FeatureEngineering(df, target_col='Pago_atiempo')
        X_train, X_test, y_train, y_test, preprocessor = fe.run_split()

        print("\n[3/4] Entrenando Modelos...")

        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, 
                random_state=42, 
                class_weight='balanced'
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150, 
                max_depth=10, 
                random_state=42, 
                class_weight='balanced'
            ),
            "XGBoost": xgb.XGBClassifier(
                use_label_encoder=False, 
                eval_metric='logloss', 
                random_state=42,
                scale_pos_weight=5
            )
        }

        for name, model in models.items():
            print(f"\n   👉 Entrenando: {name}...")
            
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            recall_class_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"       Recall (Detectar Impago): {recall_class_0:.2%}")
            print(f"       Accuracy General: {acc:.2%}")
            print("       Reporte Detallado:")
            print(classification_report(y_test, y_pred, target_names=['Impago (0)', 'Pago (1)'], zero_division=0))
            metric_to_optimize = recall_class_0 
            
            if metric_to_optimize > self.best_metric:
                self.best_metric = metric_to_optimize
                self.best_model = pipeline
                self.best_model_name = name
        print("\n" + "="*60)
        print(f"  Mejor: {self.best_model_name}")
        print(f"   Métrica (Recall Impago): {self.best_metric:.2%}")
        print("="*60)
        
        self.save_best_model()

    def save_best_model(self):
        if self.best_model:
            filename = 'src/best_model.pkl'
            joblib.dump(self.best_model, filename)
            print(f"\n💾 Modelo guardado en: {filename}")
        else:
            print(" No hay modelo para guardar.")

if __name__ == "__main__":
    PATH_EXCEL = "./Base_de_datos.xlsx"
    trainer = ModelTrainer(PATH_EXCEL)
    trainer.run_training()