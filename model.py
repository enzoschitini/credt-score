import streamlit as st

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.model_selection import train_test_split

import pickle
from io import BytesIO

st.set_page_config(layout='wide', page_title='EBAC - Projeto Final')
st.sidebar.write('# EBAC - Projeto Final')

#####################################################
#               Interface de uso
#####################################################

st.sidebar.write("## Suba o arquivo")
data_file_1 = st.sidebar.file_uploader("Bank Credit Dataset", type = ['csv','ftr'])

st.wread(data_file_1)