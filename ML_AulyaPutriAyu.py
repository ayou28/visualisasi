import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
    

st.title('Machine Learning Assignment GDGoC UNSRI')
st.write("""
Oleh  
Nama    : Aulya Putri Ayu  
NIM     : 09011182227118  
Jurusan : Sistem Komputer 2022
""")

# Load dataset
st.markdown('## Dataset')
df = pd.read_csv('used_car_dataset.csv')
df.shape
st.write("""
Data ini berisi informasi tentang mobil bekas di India, yang terdiri dari 9.582 baris data dengan 11 variabel. Data tersebut dikumpulkan hingga November 2024, sehingga memberikan gambaran menyeluruh tentang pasar mobil bekas di India.
""")
st.write("""
Dataset ini berisi variabel-variabel sebagai berikut:

- **Brand**: Produsen mobil (misalnya, Volkswagen, Maruti Suzuki, Honda, Tata)
- **Model**: Model mobil tertentu (misalnya, Taigun, Baleno, Polo, WRV)
- **Year**: Tahun pembuatan kendaraan (mulai dari model lama hingga 2024)
- **Age**: Usia kendaraan dalam tahun
- **kmDriven**: Total kilometer yang ditempuh kendaraan
- **Transmission**: Jenis transmisi (Manual atau Otomatis)
- **Owner**: Status kepemilikan (pemilik pertama atau kedua)
- **Fuel Type**: Jenis bahan bakar (Bensin, Diesel, Hibrida/CNG)
- **Posted On**: Tanggal saat iklan mobil diposting
- **Additional Info**: Detail tambahan tentang kendaraan
- **AskPrice**: Harga tercantum dalam Rupee India (₹)
""")

# Menampilkan dataset
st.markdown('## Dataset ##')
st.write('5 Baris data awal:', df.head())

# Menampilkan df.info
import io
st.markdown('## Informasi Data ##')
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)
st.write("""
Dari informasi data di atas, terlihat bahwa variabel numerik hanya terdapat pada **Year** dan **Age**. 
Untuk memudahkan pengolahan data, tipe data pada variabel **kmDriven** dan **AskPrice** akan diubah menjadi numerik.
""")

# Menghapus 'km', tanda koma ',', dan spasi yang tidak diinginkan di kmDriven
df['kmDriven'] = df['kmDriven'].replace({'km': '', ',': '', '\s+': ''}, regex=True)
df['kmDriven'] = pd.to_numeric(df['kmDriven'], errors='coerce')

# Menghapus simbol '₹' saja tanpa memengaruhi angka atau tanda koma di AskPrice
df['AskPrice'] = df['AskPrice'].str.replace('₹', '', regex=True)
df['AskPrice'] = pd.to_numeric(df['AskPrice'].replace({',': ''}, regex=True), errors='coerce')


st.markdown('## Exploration Data Analysis ##')
st.markdown('### Data Statistik ###')
st.dataframe(df.describe())
st.write("""
Dari data statistik di atas, dapat diketahui:
- Rata-rata umur mobil adalah 7.6 tahun, dengan umur mobil termuda 0 tahun dan tertua 38 tahun.
- Tahun pembuatan mobil tertua adalah tahun 1986, sedangkan tahun pembuatan terbaru adalah 2024, dengan rata-rata tahun pembuatan mobil adalah 2016.
- Harga mobil rata-rata adalah 1.063.608 rupe, dengan harga mobil termurah sebesar 15.000 rupe dan harga mobil termahal mencapai 42.500.000 rupe.
""")

# Hitung jumlah nilai hilang
st.markdown('### Missing Value ###')
st.dataframe(df.isnull().sum())
st.write('Dataset ini memiliki nilai yang hilang pada variabel **kmDriven**.')

# Menghapus nilai yang hilang dari dataset
df = df.dropna()
st.dataframe(df.isnull().sum())
st.write("Nilai yang hilang sudah dihapus dari dataset.")

# Hitung jumlah duplikat
st.markdown('### Duplicated Value ###')
duplicated_count = df.duplicated().sum()
st.write(f'Dataset ini memiliki {duplicated_count} baris data yang duplikat.')

# Tampilkan baris yang duplikat
if duplicated_count > 0:
    duplicated_rows = df[df.duplicated()]
    st.dataframe(duplicated_rows)
else:
    st.write("Tidak ditemukan baris data yang duplikat.")

# Hapus duplikat dari dataset
df = df.drop_duplicates()
st.write("Nilai duplikat telah dihapus dari dataset.")

# Menampilkan nilai yang duplikat
st.write(f'Dataset ini memiliki {df.duplicated().sum() } baris data yang duplikat.')

# menampilkan Histogram 
st.markdown('### Histogram of the Dataset ###')
df.hist(bins=20, figsize=(10, 8))
plt.tight_layout()
st.pyplot(plt)
st.write("""
Dari visualisasi histogram di atas:
- **Year**  
Sebagian besar kendaraan diproduksi antara tahun 2010 hingga 2020. Produksi kendaraan cenderung meningkat seiring waktu, dengan penurunan kecil setelah 2020.
- **Age**  
Mayoritas kendaraan memiliki usia kurang dari 10 tahun. Usia kendaraan sangat jarang melebihi 15 tahun, menunjukkan preferensi terhadap kendaraan yang relatif baru.
- **kmDriven**  
Sebagian besar kendaraan memiliki jarak tempuh di bawah 200.000 km, dengan distribusi yang sangat menurun setelahnya. Hanya sedikit kendaraan yang memiliki jarak tempuh lebih dari 500.000 km, menunjukkan bahwa kendaraan dengan jarak tempuh ekstrem cukup langka.
- **AskPrice**  
Harga kendaraan yang diminta sebagian besar berada di bawah 1 juta Rupee, dengan harga yang sangat tinggi, dan pola distribusi yang cenderung *positively skewed* (condong ke kanan).
""")


# visualisasi heatmap
st.markdown('### Correlation Heatmap of the Dataset ###')
# pilih kolom numerik
numeric_columns = df.select_dtypes(include=['number'])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.tight_layout()
st.pyplot(plt)
plt.clf()
st.write("""
Berdasarkan heatmap korelasi ini:
- Sebagian besar hubungan antar variabel memiliki koefisien korelasi yang rendah, menunjukkan korelasi lemah atau hampir tidak ada hubungan linear antara variabel-variabel tertentu.
- **Korelasi negatif yang kuat (-1)** ditemukan antara **Year** dan **Age**, yang menunjukkan bahwa semakin baru tahun produksi kendaraan, semakin rendah usia kendaraan.
- **Korelasi positif** ditemukan antara **Year** dan **AskPrice** (0.31) dan **Age** dan **kmDriven** (0.28).
- **Hubungan antara kmDriven dan AskPrice** memiliki korelasi negatif lemah (-0.14), menunjukkan bahwa jarak tempuh kendaraan memiliki sedikit dampak pada harga yang diminta.
""")

#visualisasi boxplot
st.markdown('### Boxplot Kolom Numerik ###')
plt.figure(figsize=(10, 6))
sns.boxplot(data=numeric_columns)
plt.tight_layout()
st.pyplot(plt)
plt.clf()
st.write("""
Berdasarkan boxplot ini ini:
- **Year** dan **Age** Hampir semua nilainya mirip satu sama lain, tanpa banyak nilai yang terlalu tinggi atau rendah.
- **kmDriven** memiliki distribusi yang relatif kecil, tetapi terdapat beberapa outliers. Ini menunjukkan bahwa ada beberapa kendaraan yang jarak tempuhnya jauh lebih besar dibandingkan mayoritas kendaraan lainnya.
- **AskPrice** terdapat banyak sekali outliers, menunjukkan adanya harga-harga yang sangat tinggi dibandingkan mayoritas data lainnya.
""")

# Visualisasi
st.markdown('### Hubungan antara Tahun Produksi dan Harga yang Diminta ###')
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Year', y='AskPrice', data=df)
plt.xlabel('Tahun Produksi')
plt.ylabel('Harga (Rupe)')
st.pyplot(plt)
plt.clf()
st.write("""
 **Year** dan **AskPrice** menunjukkan bahwa kendaraan yang lebih baru cenderung memiliki harga yang lebih tinggi.
""")

# Visualisasi
st.markdown('### Hubungan antara Usia Kendaraan dan Jarak Tempuh ###')
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='kmDriven', data=df)
plt.xlabel('Usia Kendaraan (Age)')
plt.ylabel('Jarak Tempuh (kmDriven)')
st.pyplot(plt)
plt.clf()
st.write("""
 **Age** dan **kmDriven**, menunjukkan bahwa kendaraan yang lebih tua cenderung memiliki jarak tempuh yang lebih tinggi.
""")

# Visualisasi
st.markdown('### Distribusi Mobil Berdasarkan Merek ###')
brand_counts = df['Brand'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=brand_counts.index, y=brand_counts.values, palette="viridis")
plt.xticks(rotation=90)
plt.xlabel("Brand")
plt.ylabel("Jumlah")
st.pyplot(plt)
plt.clf()
st.write("""
 Brand mobil terbanyak adalah **Maruti Suzuki**, diikuti dengan **Hyundai**, **Honda**, **Toyota**, **Mahindra**.
""")

# Visualisasi
st.markdown('### 20 Model Mobil Teratas ###')
model_counts = df['model'].value_counts().head(20)
plt.figure(figsize=(10, 6))
sns.barplot(y=model_counts.index, x=model_counts.values, palette="viridis")
plt.xlabel("Jumlah")
plt.ylabel("Model")
st.pyplot(plt)
plt.clf()
st.write("""
 City menjadi model mobil yang paling banyak di jual.
""")

# Visualisasi
st.markdown('### Distribusi Jenis Bahan Bakar Berdasarkan Transmisi ###')
plt.figure(figsize=(10, 6))
sns.countplot(x='FuelType', hue='Transmission', data=df, palette='viridis')
plt.xlabel('Jenis Bahan Bakar')
plt.ylabel('Jumlah Mobil')
plt.legend(title='Jenis Transmisi')
st.pyplot(plt)
plt.clf()
st.write("""
Visualisasi ini menunjukkan distribusi jenis bahan bakar berdasarkan jenis transmisi (manual dan otomatis).
- **Petrol (Bensin):** Mobil berbahan bakar bensin (petrol) lebih banyak menggunakan transmisi manual dibandingkan transmisi otomatis, dengan perbedaan jumlah yang tidak terlalu signifikan.
- **Diesel:** Mobil berbahan bakar diesel juga lebih banyak menggunakan transmisi manual, namun jumlah mobil diesel otomatis hampir setara dengan yang manual.
- **Hybrid/CNG:** Mobil berbahan bakar hybrid/CNG memiliki jumlah yang lebih rendah dibandingkan petrol dan diesel, dengan preferensi yang tetap cenderung lebih banyak menggunakan transmisi manual dibanding otomatis.

Secara keseluruhan, mobil dengan transmisi manual lebih dominan di semua jenis bahan bakar, tetapi perbedaannya lebih kecil pada jenis diesel dan hybrid/CNG dibandingkan petrol.
""")

# Visualisasi
st.markdown('### Jenis Transmisi Mobil ###')
plt.figure(figsize=(10, 6))
transmission_counts = df['Transmission'].value_counts()
plt.pie(transmission_counts.values, labels=transmission_counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set3", len(transmission_counts)))
st.pyplot(plt)
plt.clf()
st.write("""
 Manual menjadi transmisi mobil yang paling banyak di jual.
""")

# Visualisasi
st.markdown('### Distribusi Tahun Produksi Mobil ###')
year_counts = df['Year'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x=year_counts.index, y=year_counts.values, marker='o', color="blue")
plt.xlabel("Year")
plt.ylabel("Jumlah Mobil")
st.pyplot(plt)
plt.clf()
st.write("""
Distribusi tahun produksi mobil menunjukkan bahwa mobil dengan tahun produksi sekitar **2010–2015** adalah yang **paling banyak dijual**, dengan jumlah kendaraan yang tinggi. Sebaliknya, jumlah mobil yang diproduksi sejak tahun **2016 hingga sekarang** cenderung lebih **rendah**.
""")

# Visualisasi
st.markdown('### Distribusi Usia Mobil ###')
age_counts = df['Age'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
sns.barplot(x=age_counts.index, y=age_counts.values, color="purple")
plt.title("Distribusi Usia Mobil")
plt.xlabel("Age (Years)")
plt.ylabel("Jumlah Mobil")
st.pyplot(plt)
plt.clf()
st.write("""
Visualisasi ini menunjukkan bahwa **mobil dengan usia 7 tahun** merupakan yang **paling banyak ditemukan**, dengan jumlah mencapai sekitar 900-1000 unit. Setelah usia 10 tahun, jumlah mobil menurun signifikan, dan **mobil > 15 tahun jarang ditemui**. 
Hal ini menunjukkan bahwa kendaraan dengan usia yang lebih muda lebih banyak ditemukan di pasar mobil bekas. Ini mungkin disebabkan oleh kecenderungan konsumen untuk memilih mobil bekas yang relatif baru, karena kendaraan dengan usia muda cenderung memiliki performa yang lebih baik, lebih efisien dalam konsumsi bahan bakar, serta biaya perawatan yang lebih rendah dibandingkan dengan mobil yang lebih tua. Selain itu, mobil dengan usia muda masih memiliki sisa garansi dan teknologi yang lebih terkini, menjadikannya pilihan yang lebih menarik bagi pembeli.
""")

# Visualisasi
st.markdown('### Distribusi Status Kepemilikan ###')
owner_counts = df['Owner'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(owner_counts.values, labels=owner_counts.index, autopct='%1.1f%%', colors=sns.color_palette("viridis", len(owner_counts)))
st.pyplot(plt)
plt.clf()
st.write("""
Visualisasi ini menunjukkan bahwa jumlah kendaraan dengan status **kepemilikan pertama (mobil baru)** lebih besar dibandingkan dengan jumlah kendaraan dengan **pemilik kedua**. Ini menunjukkan adanya minat yang lebih tinggi terhadap kendaraan yang masih relatif baru dan dalam kondisi baik, yang sering kali dianggap lebih menarik bagi pembeli dibandingkan kendaraan yang sudah digunakan sebelumnya.
""")

st.markdown('## Kesimpulan ##')

# Insight 1: Distribusi Tahun Produksi dan Usia Kendaraan
st.write("""
### 1. Distribusi Tahun Produksi dan Usia Kendaraan
Sebagian besar mobil dalam dataset diproduksi antara tahun 2010 hingga 2020, dengan penurunan produksi yang terlihat setelah 2020. Rata-rata usia kendaraan adalah sekitar 7.6 tahun, dengan sebagian besar kendaraan memiliki usia < 10 tahun. Kendaraan yang lebih tua (> 15 tahun) jarang ditemukan, yang menunjukkan preferensi pasar terhadap mobil bekas yang relatif baru.
""")

# Insight 2: Harga Mobil dan Jarak Tempuh
st.write("""
### 2. Harga Mobil dan Jarak Tempuh
Harga mobil yang diminta bervariasi, dengan sebagian besar kendaraan memiliki harga di bawah 1 juta Rupee. Harga mobil cenderung lebih tinggi untuk kendaraan yang lebih baru, dengan beberapa mobil mencapai harga yang sangat tinggi (lebih dari 42 juta Rupee). Mobil yang lebih tua umumnya memiliki jarak tempuh yang lebih tinggi, yang menunjukkan bahwa kendaraan yang lebih banyak digunakan cenderung memiliki harga lebih rendah.
""")

# Insight 3: Korelasi Antara Tahun Produksi, Usia, dan Harga
st.write("""
### 3. Korelasi Antara Tahun Produksi, Usia, dan Harga
Terdapat korelasi negatif yang kuat antara tahun produksi dan usia kendaraan, yang menunjukkan bahwa kendaraan yang lebih baru cenderung memiliki usia yang lebih muda. Ada juga korelasi positif antara tahun produksi dan harga, yang menunjukkan bahwa mobil yang lebih baru memiliki harga yang lebih tinggi. Sementara itu, hubungan antara jarak tempuh dan harga cenderung lemah, menunjukkan bahwa jarak tempuh tidak terlalu mempengaruhi harga mobil bekas.
""")

# Insight 4: Jenis Transmisi dan Bahan Bakar
st.write("""
### 4. Jenis Transmisi dan Bahan Bakar
Mobil dengan transmisi manual lebih dominan di pasar mobil bekas, terutama untuk bahan bakar bensin dan diesel. Meskipun demikian, ada keseimbangan antara transmisi manual dan otomatis untuk mobil berbahan bakar diesel dan hybrid/CNG. Secara keseluruhan, transmisi manual lebih banyak diminati, namun ada pergeseran kecil di segmen diesel dan hybrid/CNG.
""")

# Insight 5: Distribusi Status Kepemilikan
st.write("""
### 5. Distribusi Status Kepemilikan
Mayoritas kendaraan dalam dataset adalah mobil dengan status pemilik pertama, yang menunjukkan minat yang lebih tinggi terhadap kendaraan yang masih relatif baru dan dalam kondisi baik. Hal ini mungkin dipengaruhi oleh preferensi pembeli yang menginginkan mobil bekas dengan kondisi yang lebih baik dan umur yang lebih muda.
""")

# Insight 6: Merek dan Model Mobil
st.write("""
### 6. Merek dan Model Mobil
Maruti Suzuki adalah merek yang paling banyak ditemukan dalam dataset ini, diikuti oleh merek-merek besar lainnya seperti Maruti Suzuki, Hyundai, Honda, dan Toyota. Model mobil yang paling banyak dijual adalah model dari merek tersebut, dengan City menjadi model yang paling dominan.
""")

# Insight 7: Distribusi Tahun dan Usia Mobil
st.write("""
### 7. Distribusi Tahun dan Usia Mobil
Mobil-mobil yang diproduksi sekitar 2010 hingga 2015 lebih banyak dijual, sementara jumlah mobil yang diproduksi setelah 2016 lebih sedikit. Selain itu, mobil yang lebih muda, khususnya yang berusia sekitar 7 tahun, memiliki jumlah yang lebih tinggi di pasar mobil bekas. Hal ini menunjukkan bahwa kendaraan dengan usia muda dan harga yang relatif lebih terjangkau lebih diminati oleh pembeli.
""")

st.write("""
Secara keseluruhan, pasar mobil bekas di India didominasi oleh kendaraan yang relatif baru dengan transmisi manual, harga yang bervariasi, dan preferensi terhadap mobil yang lebih muda serta berkualitas baik. Mobil dengan transmisi manual lebih dominan di semua jenis bahan bakar, tetapi perbedaannya lebih kecil pada jenis diesel dan hybrid/CNG dibandingkan dengan bahan bakar bensin.
""")









