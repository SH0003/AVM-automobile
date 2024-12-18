import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


@st.cache_resource
def load_data():
    df = pd.read_csv('classification_moteur_ma.csv')
    return df

def preprocess_data(df):
    df = df.copy()  
    features = ['Title', 'Année', 'Kilométrage', 'Boite de vitesses', 'Carburant', 'Puissance fiscale', 'Nombre de portes']
    target = 'Price'

    colonnes_numeriques = ['Price']
    for col in colonnes_numeriques:
        df[col] = df[col].fillna("").astype(str)
        df[col] = (
            df[col]
            .str.replace(' ', '', regex=False)
            .str.replace('Dhs', '', regex=False)
            .replace('', np.nan)  
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['Price'], inplace=True) 
    df['Price'] = df['Price'].astype(float)  
    df.dropna(subset=features + [target], inplace=True)  
    
    X = df[features]
    y = df[target]

    categorical_cols = ['Boite de vitesses', 'Carburant', 'Title']
    numerical_cols = ['Année', 'Kilométrage', 'Puissance fiscale', 'Nombre de portes']

    _preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, _preprocessor

@st.cache_resource
def train_model(X_train, y_train, _preprocessor):
    model = Pipeline(steps=[('preprocessor', _preprocessor),
                            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
    model.fit(X_train, y_train)
    return model

st.title("AUTO VISION \n Votre Application de Prédiction des Prix des Voitures")
st.write("Cette application permet de prédire le prix d'une voiture en fonction de ses caractéristiques.")


df = load_data()

if st.checkbox('Afficher les données brutes'):
    st.write(df.sample(5)) 

X_train, X_test, y_train, y_test, _preprocessor = preprocess_data(df)


model = train_model(X_train, y_train,_preprocessor)

st.sidebar.header('Paramètres de la voiture')

annee_min = int(df['Année'].min()) if 'Année' in df.columns else 1990
annee_max = int(df['Année'].max()) if 'Année' in df.columns else 2024
title = st.sidebar.selectbox('Title', df['Title'].unique())
annee = st.sidebar.slider('Année', min_value=1990, max_value=annee_max, value=2015)
kilometrage = st.sidebar.number_input('Kilométrage (km)', min_value=0, max_value=500000, value=50000, step=1, format="%d")
boite_vitesses = st.sidebar.selectbox('Boite de vitesses', df['Boite de vitesses'].unique())
carburant = st.sidebar.selectbox('Carburant', df['Carburant'].unique())
puissance_fiscale = st.sidebar.number_input('Puissance fiscale', min_value=1, max_value=50, value=6, step=1, format="%d")
nombre_portes = st.sidebar.number_input('Nombre de portes', min_value=2, max_value=5, value=4, step=1, format="%d")

input_data = pd.DataFrame({
    'Title': [title],
    'Année': [annee],
    'Kilométrage': [kilometrage],
    'Boite de vitesses': [boite_vitesses],
    'Carburant': [carburant],
    'Puissance fiscale': [puissance_fiscale],
    'Nombre de portes': [nombre_portes]
})

try:
    prediction = model.predict(input_data)
    st.write(f'**Prix prédit : {prediction[0]:.2f} Dhs**')
except Exception as e:
    st.write(f"Erreur dans la prédiction : {e}")
