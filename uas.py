import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Menampilkan judul
app_title = st.text_input('Masukkan judul aplikasi:')
st.title(app_title)

# Memasukkan jumlah data yang akan dimasukkan
num_data = st.number_input('Masukkan jumlah data:', min_value=2, step=1)

# Inisialisasi list kosong untuk menyimpan data
data = {'X': [], 'Y': []}

# Memasukkan data
for i in range(num_data):
    st.write(f'Data ke-{i+1}:')
    x = st.number_input(f'Masukkan nilai X-{i+1}:')
    y = st.number_input(f'Masukkan nilai Y-{i+1}:')
    data['X'].append(x)
    data['Y'].append(y)

# Membuat dataframe dari data yang dimasukkan
df = pd.DataFrame(data)

# Menampilkan tabel data
st.write('Data yang Dimasukkan:')
st.write(df)

# Memilih variabel independen dan dependen
x_label_custom = st.text_input('Masukkan label untuk sumbu X (opsional):', 'X')
y_label_custom = st.text_input('Masukkan label untuk sumbu Y (opsional):', 'Y')

if x_label_custom.strip() == '':
    x_label = 'X'
else:
    x_label = x_label_custom.strip()

if y_label_custom.strip() == '':
    y_label = 'Y'
else:
    y_label = y_label_custom.strip()

x = np.array(df[x_label]).reshape(-1, 1)
y = np.array(df[y_label])

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
st.write('R-squared:', r_squared)

# Menampilkan persamaan regresi linear
slope = model.coef_[0]
intercept = model.intercept_
st.write('Persamaan Regresi Linear:')
st.write(f'{y_label} = {slope} * {x_label} + {intercept}')

# Tombol untuk menampilkan plot regresi, persamaan, dan R-squared
if st.button('Tampilkan Grafik, Persamaan, dan R-squared'):
    # Menampilkan plot regresi
    st.write('Plot Regresi:')
    plt.scatter(x, y, color='blue')
    plt.plot(x, model.predict(x), color='red')
    plt.title('Linear Regression')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    st.pyplot(plt)

    # Menampilkan tabel persamaan regresi dan R-squared
    st.write('Tabel Persamaan Regresi dan R-squared:')
    result_df = pd.DataFrame({'Koefisien': [model.intercept_, model.coef_[0], r_squared]},
                             index=['Intercept', 'Slope', 'R-squared'])
    st.write(result_df)
