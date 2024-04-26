import pandas as pd
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Menampilkan judul
st.title('kalkulator ketelitian pengukuran pada data praktikum')
app_title = st.text_input('Masukkan judul grafik:')
st.title(app_title)

#masukkan nama variabel
x_nama = st.text_input('Masukkan variabel manipulasi (x)')
y_nama = st.text_input('Masukkan variabel respon (y)')

# Memasukkan jumlah data yang akan dimasukkan
num_data = st.number_input('Masukkan jumlah data:', min_value=2, step=1)

# Inisialisasi list kosong untuk menyimpan data
data = {x_nama: [], y_nama: []}

# Memasukkan data
for i in range(num_data):
    st.write(f'Data ke-{i+1}:')
    x = st.number_input(f'Masukkan data x {i+1}:')
    y = st.number_input(f'Masukkan data y {i+1}:')
    data[x_nama].append(x)
    data[y_nama].append(y)

# Membuat dataframe dari data yang dimasukkan
df = pd.DataFrame(data)

# Menampilkan tabel data
st.write('Data yang Dimasukkan:')
st.write(df)

# Memilih variabel independen dan dependen

x_nilai = df[x_nama]
y_nilai = df[y_nama]

x = np.array(x_nilai)
y = np.array(y_nilai)
x = x.reshape(len(x),1)
y = y.reshape(len(y),1)

# Membuat model regresi linear
model = LinearRegression()
model.fit(x, y)

# Prediksi nilai y berdasarkan model
y_pred = model.predict(x)

# Menghitung R-squared
r_squared = r2_score(y, y_pred)

# Menampilkan koefisien regresi dan R-squared
st.write('Koefisien Regresi:')
st.write('Intercept:', model.intercept_)
st.write('Slope:', model.coef_[0])

# Menampilkan persamaan regresi linear
slope = model.coef_[0]
intercept = model.intercept_



# Tombol untuk menampilkan plot regresi, persamaan, dan R-squared
if st.button('Tampilkan Grafik, Persamaan, dan R-squared'):
    # Menampilkan plot regresi
    st.write('Grafik:')
    plt.scatter(x, y, color='blue', label ='data')
    plt.plot(x, model.predict(x), color='red', label='linear regression')
    plt.title(app_title)
    plt.xlabel(x_nama)
    plt.ylabel(y_nama)
    plt.legend()
    st.pyplot(plt)

    # Menampilkan persamaan dan R squared
    st.write('persamaan linear dan R²:')
    st.write('=================================')
    st.write(f'y = {slope}x + {intercept}')
    st.write('R²:',r_squared)
    st.write('=================================')
    st.balloons()
