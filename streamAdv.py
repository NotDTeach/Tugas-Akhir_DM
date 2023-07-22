import pickle
import streamlit as st

# Baca Model
models = pickle.load(open('deployment.sav', 'rb'))


# Judul Web
st.title("Prediksi Pendapatan Iklan")

TV = st.number_input("Masukkan Nilai Koefisien Beta-1(β1) : ")
Radio = st.number_input("Masukkan Nilai Koefisien Beta-2(β2) : ")
Newspaper = st.number_input("Masukkan Nilai Koefisien Beta-3(β3) : ")


# Code Prediksi
pred = TV + Radio + Newspaper

# Membuat Tombol Hasil
if st.button('Test Prediksi'): pred = models.predict([[TV,Radio,Newspaper]])

if (pred == 0):
    pred = 'Prediksi Multiple Linear Regression Kurang Baik, Silahkan Gunakan Algoritma Lain !', pred
else :
    pred = 'Prediksi Multiple Linear Regression Baik !', pred
st.success(pred)