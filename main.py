import streamlit as st
import yfinance as yf
import pandas as pd

st.title("Hello")
st.header("Header1")
st.text("test")
df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df