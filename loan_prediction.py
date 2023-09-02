import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from time import time
import numpy as np


def main():
    df_bruto = pd.read_csv("loan_approval_dataset.csv")
    df_processado = pre_processar_dados(df_bruto)
    df_transformado = transformar_dados(df_processado)
    resultados = cross_validation(df_processado, df_transformado)
    construir_dataframe_de_resultados(resultados)
    df_transformado.to_csv("dataset_transformado.csv")


def pre_processar_dados(df):
    # Remoção dos espaços em branco do nome das colunas
    df.rename(columns=lambda x: x.strip(), inplace=True)

    # Remoção da coluna 'loan_id', pois é apenas o identificador único do empréstimo e não impacta no resultado dos modelos
    df.drop("loan_id", axis=1, inplace=True)

    # Conversão das variáveis categóricas para valor numérico
    df['education'].replace({' Not Graduate': 0, ' Graduate': 1}, inplace=True)
    df['self_employed'].replace({' No': 0, ' Yes': 1}, inplace=True)
    df['loan_status'].replace({' Rejected': 0, ' Approved': 1}, inplace=True)

    return df


def transformar_dados(df):
    # Remoção de outliers pela distância interquartil
    cols = ['income_annum', 'loan_amount', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
    q1 = df[cols].quantile(0.25)
    q3 = df[cols].quantile(0.75)
    iqr = q3 - q1
    filter_conditions = ((df[cols] < q1 - 1.5 * iqr) | (df[cols] > q3 + 1.5 * iqr)).any(axis=1)
    df = df[~filter_conditions]

    # Balanceamento dos dados usando a técnica SMOTE
    y = df['loan_status']
    x = df.drop('loan_status', axis = 1)
    X_res, y_res = SMOTE().fit_resample(x, y)

    # Padronização dos dados com a técnica StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    # Criação do dataset transformado
    df_final = pd.DataFrame(X_scaled, columns=X_res.columns)
    df_final['loan_status'] = y_res

    return df_final


def cross_validation(df_bruto, df_transformado):
    # Datasets
    datasets = {
        'df_bruto': df_bruto, 
        'df_transformado': df_transformado
    }

    # Criação dos modelos
    modelos = {
        'Árvore de Decisão': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=25, random_state=42),
        'SVM': SVC(kernel='rbf', C=1.0, random_state=42),
        'Naive Bayes': GaussianNB(),
    }
    metricas = ['accuracy', 'precision', 'recall'] # Métricas de avaliação que serão utilizadas
    resultados_modelos = []

    # Para cada dataset
    for nome_dataset, dataset in datasets.items():
        features, rotulos = dataset.drop('loan_status', axis = 1), dataset['loan_status']
        # Para cada modelo
        for nome_modelo, modelo in modelos.items():
            # Estruturas para armazenar os resultados de cada modelo em cada execução do cross-validation
            tempos_execucao = []
            resultados_metricas = {metrica: [] for metrica in metricas}
            for _ in range(5):
                # Execução do cross-validation e cálculo do tempo de execução
                inicio = time()
                resultados_cv = cross_validate(modelo, features, rotulos, cv=10, scoring=metricas)
                fim = time()
                tempos_execucao.append(fim - inicio)

                # Adicionando o resultado de cada métrica no mapa
                for metrica in metricas:
                    resultados_metricas[metrica].append(resultados_cv[f'test_{metrica}'])
            
            # Adicionando resultados médios de cada modelo na lista de resultados
            resultados_medios_modelo = {'Dataset': nome_dataset, 'Modelo': nome_modelo, 'Tempo Médio': np.mean(tempos_execucao)}
            for metrica, resultado in resultados_metricas.items():
                resultados_medios_modelo[metrica] = np.mean(resultado)
            resultados_modelos.append(resultados_medios_modelo)

    return resultados_modelos


def construir_dataframe_de_resultados(resultados):
    df = pd.DataFrame(resultados)
    df.rename(columns={
        'accuracy': 'Acurácia Média', 
        'precision': 'Precisão Média',
        'recall': 'Sensibilidade Média',
    }, inplace=True)
    print(df)


main()