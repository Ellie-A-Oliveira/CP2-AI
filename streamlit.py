import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/games.csv')
x = df[['Release date', 'Estimated owners', 'Peak CCU', 'Required age', 'DLC count', 'Average playtime two weeks']]
x['Release date'] = pd.to_numeric(pd.to_datetime(x['Release date'], errors='coerce'))
x['Estimated owners'] = x['Estimated owners'].apply(lambda x: x.split(' - ')[1])
y = df['Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train)

st.write(gbr.score(x_test, y_test))