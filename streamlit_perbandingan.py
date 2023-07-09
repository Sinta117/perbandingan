import streamlit as st
import pandas as pd
import numpy as np
import json

import perbandingan_hoax

st.write("""
# Fake Political News Detection
""")

w2i, i2w = {'Hoax': 0, 'Non Hoax': 1}, {0: 'Hoax', 1: 'Non Hoax'}

input = st.text_input('Contoh Teks Judul Berita Politik', '')

col1, col2, col3 = st.columns(3)
click = col1.button("LSTM")
tekan = col2.button("XGBoost")
push = col3.button("RandomForest")

disp1, disp2, disp3 = st.columns(3)

if click:
    label_LSTM = perbandingan_hoax.predict_model_LSTM(input)
    if int(label_LSTM.round())==0:
        st.markdown("""
        <style>
        div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 1% 1% 1% 1%;
        border-radius: 2px;
        color: rgb(255, 0, 0);
        overflow-wrap: break-word;
        }

        /* breakline for metric text         */
        div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
        overflow-wrap: break-word;
        white-space: break-spaces;
        color: red;
        font-size:200%;
        }
        </style>
        """
        , unsafe_allow_html=True)
        score = 100-round(label_LSTM[0][0]*100, 2)
    else:
        st.markdown("""
        <style>
        div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 1% 1% 1% 1%;
        border-radius: 2px;
        color: rgb(13, 252, 13);
        overflow-wrap: break-word;
        }

        /* breakline for metric text         */
        div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
        overflow-wrap: break-word;
        white-space: break-spaces;
        color: green;
        font-size:200%;
        }
        </style>
        """
        , unsafe_allow_html=True)
        score = round(label_LSTM[0][0]*100, 2)

    disp2.metric(i2w[int(label_LSTM.round())], '{}%'.format(score))

if tekan:
    label_XGB = perbandingan_hoax.predict_model_xgb(input)
    if int(label_XGB.round())==0:
        st.markdown("""
        <style>
        div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 1% 1% 1% 1%;
        border-radius: 2px;
        color: rgb(255, 0, 0);
        overflow-wrap: break-word;
        }

        /* breakline for metric text         */
        div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
        overflow-wrap: break-word;
        white-space: break-spaces;
        color: red;
        font-size:200%;
        }
        </style>
        """
        , unsafe_allow_html=True)
        score = 100-round(label_XGB[0]*100, 2)
    else:
        st.markdown("""
        <style>
        div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 1% 1% 1% 1%;
        border-radius: 2px;
        color: rgb(13, 252, 13);
        overflow-wrap: break-word;
        }

        /* breakline for metric text         */
        div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
        overflow-wrap: break-word;
        white-space: break-spaces;
        color: green;
        font-size:200%;
        }
        </style>
        """
        , unsafe_allow_html=True)
        score = round(label_XGB[0]*100, 2)

    disp2.metric(i2w[int(label_XGB.round())], '{}%'.format(score))


if push:
    label_RF = perbandingan_hoax.predict_model_rf(input)
    if label_RF[0].argmax()==0:
        st.markdown("""
            <style>
            div[data-testid="metric-container"] {
            background-color: rgba(28, 131, 225, 0.1);
            border: 1px solid rgba(28, 131, 225, 0.1);
            padding: 1% 1% 1% 1%;
            border-radius: 2px;
            color: rgb(255, 0, 0);
            overflow-wrap: break-word;
            }

            /* breakline for metric text         */
            div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
            overflow-wrap: break-word;
            white-space: break-spaces;
            color: red;
            font-size:200%;
            }
            </style>
            """
            , unsafe_allow_html=True)

    else:
        st.markdown("""
            <style>
            div[data-testid="metric-container"] {
            background-color: rgba(28, 131, 225, 0.1);
            border: 1px solid rgba(28, 131, 225, 0.1);
            padding: 1% 1% 1% 1%;
            border-radius: 2px;
            color: rgb(13, 252, 13);
            overflow-wrap: break-word;
            }

            /* breakline for metric text         */
            div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
            overflow-wrap: break-word;
            white-space: break-spaces;
            color: green;
            font-size:200%;
            }
            </style>
            """
            , unsafe_allow_html=True)

    score = label_RF[0].max()

    disp2.metric(i2w[label_RF[0].argmax()], '{}%'.format(score*100))

