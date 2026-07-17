# Menulis seluruh kode berikut ke dalam file bernama app.py.

import streamlit as st                                                # Mengimpor library Streamlit untuk membuat aplikasi web interaktif.
import pandas as pd                                                   # Mengimpor library pandas untuk pengolahan data tabel.
import numpy as np                                                    # Mengimpor library numpy untuk operasi numerik dan array.
import pickle                                                         # Mengimpor library pickle untuk menyimpan dan memuat objek Python.
import skfuzzy as fuzz                                                # Mengimpor library skfuzzy untuk logika fuzzy.
from skfuzzy import control as ctrl                                   # Mengimpor modul kontrol fuzzy dari skfuzzy.
from sklearn.metrics.pairwise import cosine_similarity                # Mengimpor fungsi cosine similarity untuk menghitung kemiripan data.
from sklearn.preprocessing import MinMaxScaler                        # Mengimpor MinMaxScaler untuk normalisasi data.
import re                                                             # Mengimpor library regular expression untuk manipulasi pola teks.
from collections import Counter                                       # Mengimpor Counter untuk menghitung frekuensi data.

# --- Custom CSS for white background --- 
st.markdown("""
<style>
.stApp { 
  background-color: white;
}
.st-emotion-cache-zt5ig8 { /* Targeting the main content area */
    background-color: white;
}
.st-emotion-cache-gh2jqd { /* Targeting the sidebar content area */
    background-color: white;
}
</style>
""", unsafe_allow_html=True)

# --- 1. Load Model Components and Data ---
@st.cache_resource                                                    # Menyimpan hasil fungsi ke cache agar tidak dimuat ulang setiap aplikasi dijalankan.
def load_model_components():                                          # Membuat fungsi untuk memuat seluruh komponen model.
    df_loaded = pd.read_pickle('processed_df.pkl')                    # Memuat DataFrame hasil preprocessing dari file pickle.
    similarity_matrix_loaded = np.load('similarity_matrix.npy')       # Memuat similarity matrix dari file numpy.

    with open('model_params.pkl', 'rb') as f:                         # Membuka file parameter model dalam mode baca biner.
        model_params_loaded = pickle.load(f)                          # Memuat parameter model dari file pickle.

    with open('scaler.pkl', 'rb') as f:                               # Membuka file scaler dalam mode baca biner.
        scaler_loaded = pickle.load(f)                                # Memuat objek scaler dari file pickle.

    # Reconstruct fitur_final from the loaded df_loaded to be used for CBF against new data
    df_kondisi_loaded = df_loaded['Kondisi Medis'].fillna('').str.get_dummies(sep=',') # Mengubah kondisi medis menjadi fitur one-hot encoding.
    df_alergi_loaded = df_loaded['Alergi Makanan'].fillna('').str.get_dummies(sep=',') # Mengubah alergi makanan menjadi fitur one-hot encoding.

    fitur_numerik_loaded = df_loaded[['IMT','Usia','Lingkar Perut (cm)', # Mengambil fitur numerik utama untuk model.
                                      'Kadar Lemak Tubuh (%)','Kadar Kolesterol (mg/dL)',
                                      'Aktivitas_enc', 'Berat Badan (kg)', 'Tinggi Badan (m)',
                                      'Denyut Jantung (bpm)', 'Kebutuhan Kalori (kkal)']].copy() # Using .copy() to avoid SettingWithCopyWarning

    fitur_final_loaded = pd.concat([fitur_numerik_loaded, df_kondisi_loaded, df_alergi_loaded], axis=1) # Menggabungkan seluruh fitur numerik dan kategorikal.
    fitur_final_loaded.fillna(0, inplace=True)                 # Mengisi nilai kosong dengan angka 0.

    return df_loaded, similarity_matrix_loaded, model_params_loaded, scaler_loaded, fitur_final_loaded # Mengembalikan seluruh komponen model.

df, similarity_matrix, model_params, scaler, fitur_final_training = load_model_components() # Memuat seluruh komponen model ke variabel utama.

THRESHOLD_1 = model_params['THRESHOLD_1']                      # Mengambil threshold pertama dari parameter model.
THRESHOLD_2 = model_params['THRESHOLD_2']                      # Mengambil threshold kedua dari parameter model.
BEST_W_FUZZY = model_params['BEST_W_FUZZY']                    # Mengambil bobot terbaik metode fuzzy.
BEST_W_CBF = model_params['BEST_W_CBF']                        # Mengambil bobot terbaik metode CBF.
label_map = model_params['label_map']                          # Mengambil mapping label kategori ke numerik.
inverse_label_map = model_params['inverse_label_map']          # Mengambil mapping numerik ke kategori.

# --- 2. Fuzzy Logic System (Re-definition for Streamlit) ---
# Define Fuzzy variables
imt = ctrl.Antecedent(np.arange(10, 55, 1), 'imt')             # Membuat variabel input fuzzy IMT.
lingkar = ctrl.Antecedent(np.arange(60, 140, 1), 'lingkar')    # Membuat variabel input fuzzy lingkar perut.
lemak = ctrl.Antecedent(np.arange(5, 45, 1), 'lemak')          # Membuat variabel input fuzzy kadar lemak tubuh.
aktivitas = ctrl.Antecedent(np.arange(1, 4, 0.5), 'aktivitas') # Membuat variabel input fuzzy aktivitas fisik.
tekanan = ctrl.Antecedent(np.arange(80, 200, 1), 'tekanan')    # Membuat variabel input fuzzy tekanan darah.
kolesterol = ctrl.Antecedent(np.arange(100, 310, 1), 'kolesterol') # Membuat variabel input fuzzy kolesterol.
kondisi = ctrl.Antecedent(np.arange(0, 11, 1), 'kondisi')      # Membuat variabel input fuzzy jumlah kondisi medis.
diet = ctrl.Consequent(np.arange(0, 101, 1), 'diet')           # Membuat variabel output fuzzy kategori diet.

# Membership functions
imt['rendah']    = fuzz.trimf(imt.universe, [10, 18, 24])      # Membuat membership function IMT rendah.
imt['normal']    = fuzz.trimf(imt.universe, [22, 25, 28])      # Membuat membership function IMT normal.
imt['tinggi']    = fuzz.trimf(imt.universe, [27, 33, 40])      # Membuat membership function IMT tinggi.
imt['obesitas']  = fuzz.trimf(imt.universe, [35, 45, 55])      # Membuat membership function IMT obesitas.

lingkar['normal'] = fuzz.trimf(lingkar.universe, [60, 75, 90]) # Membuat membership function lingkar perut normal.
lingkar['tinggi'] = fuzz.trimf(lingkar.universe, [85, 105, 140]) # Membuat membership function lingkar perut tinggi.

lemak['rendah'] = fuzz.trimf(lemak.universe, [5, 12, 20])      # Membuat membership function lemak rendah.
lemak['normal'] = fuzz.trimf(lemak.universe, [18, 24, 30])     # Membuat membership function lemak normal.
lemak['tinggi'] = fuzz.trimf(lemak.universe, [28, 35, 45])     # Membuat membership function lemak tinggi.

aktivitas['rendah'] = fuzz.trimf(aktivitas.universe, [1, 1, 1.5]) # Membuat membership function aktivitas rendah.
aktivitas['sedang'] = fuzz.trimf(aktivitas.universe, [1.5, 2, 2.5]) # Membuat membership function aktivitas sedang.
aktivitas['tinggi'] = fuzz.trimf(aktivitas.universe, [2, 3, 3]) # Membuat membership function aktivitas tinggi.

tekanan['normal'] = fuzz.trimf(tekanan.universe, [80, 110, 130]) # Membuat membership function tekanan normal.
tekanan['pra_tinggi'] = fuzz.trimf(tekanan.universe, [125, 140, 150]) # Membuat membership function tekanan pra-tinggi.
tekanan['tinggi'] = fuzz.trimf(tekanan.universe, [145, 170, 200]) # Membuat membership function tekanan tinggi.

kolesterol['normal'] = fuzz.trimf(kolesterol.universe, [100, 160, 200]) # Membuat membership function kolesterol normal.
kolesterol['tinggi'] = fuzz.trimf(kolesterol.universe, [190, 250, 310]) # Membuat membership function kolesterol tinggi.

kondisi['ringan'] = fuzz.trimf(kondisi.universe, [0, 1, 2])     # Membuat membership function kondisi medis ringan.
kondisi['sedang'] = fuzz.trimf(kondisi.universe, [1, 3, 5])     # Membuat membership function kondisi medis sedang.
kondisi['berat'] = fuzz.trimf(kondisi.universe, [4, 7, 10])     # Membuat membership function kondisi medis berat.

diet['ringan'] = fuzz.trimf(diet.universe, [0, 20, 40])         # Membuat membership function output diet ringan.
diet['sedang'] = fuzz.trimf(diet.universe, [30, 50, 70])        # Membuat membership function output diet sedang.
diet['ketat'] = fuzz.trimf(diet.universe, [60, 80, 100])        # Membuat membership function output diet ketat.

# Fuzzy Rules
rules = []                                                           # Membuat list kosong untuk menyimpan seluruh aturan fuzzy.
rules.append(ctrl.Rule(imt['rendah'] & aktivitas['tinggi'], diet['ringan'])) # Jika IMT rendah dan aktivitas tinggi maka diet ringan.
rules.append(ctrl.Rule(imt['normal'] & aktivitas['tinggi'] & tekanan['normal'], diet['ringan'])) # Jika IMT normal, aktivitas tinggi, dan tekanan normal maka diet ringan.
rules.append(ctrl.Rule(imt['normal'] & kolesterol['normal'] & kondisi['ringan'], diet['ringan'])) # Jika IMT normal, kolesterol normal, dan kondisi medis ringan maka diet ringan.
rules.append(ctrl.Rule(imt['rendah'] & lemak['rendah'], diet['ringan'])) # Jika IMT rendah dan lemak rendah maka diet ringan.
rules.append(ctrl.Rule(imt['normal'], diet['sedang']))               # Jika IMT normal maka diet sedang.
rules.append(ctrl.Rule(imt['tinggi'] & aktivitas['tinggi'], diet['sedang'])) # Jika IMT tinggi dan aktivitas tinggi maka diet sedang.
rules.append(ctrl.Rule(imt['tinggi'] & lemak['normal'], diet['sedang'])) # Jika IMT tinggi dan lemak normal maka diet sedang.
rules.append(ctrl.Rule(imt['obesitas'] & aktivitas['tinggi'] & tekanan['normal'], diet['sedang'])) # Jika obesitas tetapi aktivitas tinggi dan tekanan normal maka diet sedang.
rules.append(ctrl.Rule(imt['normal'] & kolesterol['tinggi'], diet['sedang'])) # Jika IMT normal namun kolesterol tinggi maka diet sedang.
rules.append(ctrl.Rule(aktivitas['sedang'] & tekanan['pra_tinggi'], diet['sedang'])) # Jika aktivitas sedang dan tekanan pra-tinggi maka diet sedang.
rules.append(ctrl.Rule(imt['obesitas'], diet['ketat']))              # Jika IMT obesitas maka diet ketat.
rules.append(ctrl.Rule(imt['tinggi'] & aktivitas['rendah'], diet['ketat'])) # Jika IMT tinggi dan aktivitas rendah maka diet ketat.
rules.append(ctrl.Rule(kolesterol['tinggi'] & lemak['tinggi'], diet['ketat'])) # Jika kolesterol tinggi dan lemak tinggi maka diet ketat.
rules.append(ctrl.Rule(tekanan['tinggi'] & kondisi['berat'], diet['ketat'])) # Jika tekanan darah tinggi dan kondisi medis berat maka diet ketat.
rules.append(ctrl.Rule(imt['obesitas'] & tekanan['tinggi'], diet['ketat'])) # Jika obesitas dan tekanan darah tinggi maka diet ketat.
rules.append(ctrl.Rule(kondisi['berat'] & lemak['tinggi'], diet['ketat'])) # Jika kondisi medis berat dan lemak tinggi maka diet ketat.
rules.append(ctrl.Rule(lingkar['tinggi'] & imt['obesitas'], diet['ketat'])) # Jika lingkar perut tinggi dan obesitas maka diet ketat.
rules.append(ctrl.Rule(imt['obesitas'] & kolesterol['tinggi'], diet['ketat'])) # Jika obesitas dan kolesterol tinggi maka diet ketat.
rules.append(ctrl.Rule(imt['normal'] | imt['tinggi'] | imt['obesitas'], diet['sedang'])) # Aturan default untuk IMT normal, tinggi, atau obesitas menjadi diet sedang.
diet_ctrl = ctrl.ControlSystem(rules)                                # Membuat sistem kontrol fuzzy berdasarkan seluruh aturan.
diet_sim = ctrl.ControlSystemSimulation(diet_ctrl)                   # Membuat simulator fuzzy untuk melakukan prediksi.

# --- 3. Prediction Functions (from notebook) ---
def hitung_kondisi_medis(kondisi_str):                               # Membuat fungsi untuk menghitung jumlah kondisi medis pasien.
    if not isinstance(kondisi_str, str):                             # Mengecek apakah data kondisi medis bukan string.
        return 0                                                     # Mengembalikan nilai 0 jika bukan string.
    kondisi_list = kondisi_str.split(',')                            # Memisahkan kondisi medis berdasarkan tanda koma.
    kondisi_list = [k.strip() for k in kondisi_list if k.strip()]    # Menghapus spasi kosong dan data kosong dari list.
    return min(len(kondisi_list), 10)                                # Mengembalikan jumlah kondisi medis maksimal 10.

def extract_systolic(tekanan_str):                                   # Membuat fungsi untuk mengambil tekanan sistolik dari format tekanan darah.
    match = re.match(r'(\d+)/', str(tekanan_str))                    # Mencari angka sebelum tanda "/" menggunakan regex.
    if match:                                                        # Mengecek apakah pola tekanan darah ditemukan.
        return int(match.group(1))                                   # Mengembalikan angka sistolik sebagai integer.
    return 120                                                       # Mengembalikan nilai default 120 jika format tidak valid.

def prediksi_fuzzy(input_data):                                      # Membuat fungsi untuk melakukan prediksi fuzzy.
    try:                                                             # Memulai blok percobaan untuk menghindari error.
        diet_sim.input['imt'] = input_data['IMT']                    # Memasukkan nilai IMT ke sistem fuzzy.
        diet_sim.input['lingkar'] = input_data['Lingkar Perut (cm)'] # Memasukkan nilai lingkar perut ke sistem fuzzy.
        diet_sim.input['lemak'] = input_data['Kadar Lemak Tubuh (%)'] # Memasukkan kadar lemak tubuh ke sistem fuzzy.
        diet_sim.input['aktivitas'] = input_data['Aktivitas_enc']    # Memasukkan nilai aktivitas fisik ke sistem fuzzy.
        diet_sim.input['tekanan'] = extract_systolic(input_data['Tekanan Darah']) # Memasukkan tekanan sistolik ke sistem fuzzy.
        diet_sim.input['kolesterol'] = input_data['Kadar Kolesterol (mg/dL)'] # Memasukkan kadar kolesterol ke sistem fuzzy.
        diet_sim.input['kondisi'] = hitung_kondisi_medis(input_data['Kondisi Medis']) # Memasukkan jumlah kondisi medis ke sistem fuzzy.
        diet_sim.compute()                                           # Menjalankan proses inferensi fuzzy.
        return diet_sim.output['diet']                               # Mengembalikan hasil skor output fuzzy.
    except Exception as e:                                           # Menangkap error jika proses fuzzy gagal.
        # Fallback to IMT-based default if fuzzy computation fails
        if input_data['IMT'] > 30:                                   # Mengecek apakah IMT lebih dari 30.
            return 80.0                                              # Mengembalikan skor diet ketat.
        elif input_data['IMT'] > 25:                                 # Mengecek apakah IMT lebih dari 25.
            return 50.0                                              # Mengembalikan skor diet sedang.
        else:                                                        # Kondisi jika IMT rendah atau normal.
            return 20.0                                              # Mengembalikan skor diet ringan.

def kategori_diet(score):                                            # Membuat fungsi untuk mengubah skor fuzzy menjadi kategori diet.
    if score < THRESHOLD_1:                                          # Mengecek apakah skor di bawah threshold pertama.
        return 'Ringan'                                              # Mengembalikan kategori diet ringan.
    elif score < THRESHOLD_2:                                        # Mengecek apakah skor di bawah threshold kedua.
        return 'Sedang'                                              # Mengembalikan kategori diet sedang.
    else:                                                            # Kondisi jika skor lebih besar atau sama dengan threshold kedua.
        return 'Ketat'                                               # Mengembalikan kategori diet ketat.

def rekomendasi_cbf(index_pasien, top_n=5):                          # Membuat fungsi rekomendasi menggunakan Content-Based Filtering.
    sim_scores = list(enumerate(similarity_matrix[index_pasien]))    # Mengambil similarity score pasien dan mengubahnya menjadi list.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) # Mengurutkan similarity score dari terbesar ke terkecil.
    sim_scores = sim_scores[1:top_n+1]                               # Mengambil top N pasien paling mirip selain dirinya sendiri.
    hasil = []                                                       # Membuat list kosong untuk menyimpan hasil rekomendasi.
    for i, score in sim_scores:                                      # Melakukan perulangan untuk setiap pasien mirip.
        hasil.append({                                               # Menambahkan data pasien mirip ke list hasil.
            'Index': i,                                              # Menyimpan index pasien.
            'Nama': df.iloc[i]['Nama Lengkap'],                      # Menyimpan nama pasien.
            'Similarity': round(score, 4),                           # Menyimpan similarity score yang dibulatkan 4 digit.
            'Label': df.iloc[i]['Label Diet']                        # Menyimpan label diet pasien.
        })
    return hasil                                                     # Mengembalikan daftar rekomendasi pasien mirip.

def get_cbf_majority_vote(index_pasien, top_n=5):                    # Membuat fungsi untuk menentukan label diet mayoritas dari pasien yang mirip.
    similar = rekomendasi_cbf(index_pasien, top_n)                   # Mengambil daftar pasien paling mirip menggunakan metode CBF.
    labels = [s['Label'] for s in similar]                           # Mengambil seluruh label diet dari pasien mirip.
    if not labels:                                                   # Mengecek apakah daftar label kosong.
        return 'Sedang'                                              # Mengembalikan label default 'Sedang' jika tidak ada data.
    vote_count = Counter(labels)                                     # Menghitung jumlah kemunculan setiap label diet.
    return vote_count.most_common(1)[0][0]                           # Mengembalikan label dengan jumlah terbanyak.

def prediksi_ensemble(index_pasien, w_fuzzy, w_cbf):                 # Membuat fungsi prediksi ensemble menggunakan fuzzy dan CBF.
    row = df.iloc[index_pasien]                                      # Mengambil data pasien berdasarkan index.
    fuzzy_score = prediksi_fuzzy(row)                                # Menghitung skor fuzzy pasien.
    fuzzy_label = kategori_diet(fuzzy_score)                         # Mengubah skor fuzzy menjadi kategori diet.
    fuzzy_enc = label_map[fuzzy_label]                               # Mengubah label fuzzy menjadi nilai numerik.
    cbf_label = get_cbf_majority_vote(index_pasien, top_n=5)         # Mengambil label hasil voting CBF.
    cbf_enc = label_map[cbf_label]                                   # Mengubah label CBF menjadi nilai numerik.

    ensemble_score = w_fuzzy * fuzzy_enc + w_cbf * cbf_enc           # Menghitung skor ensemble berdasarkan bobot fuzzy dan CBF.
    if ensemble_score < 0.8:                                         # Mengecek apakah skor ensemble rendah.
            return 0                                                     # Mengatur hasil prediksi menjadi diet ringan.
    elif ensemble_score < 1.8:                                 # Mengecek apakah skor ensemble sedang.
            return 1                                                   # Mengatur hasil prediksi menjadi diet sedang.
    else:                                                          # Kondisi jika skor ensemble tinggi.
            return 2                                                   # Mengatur hasil prediksi menjadi diet ketat.

def prediksi_final_app(index_pasien):                                # Membuat fungsi prediksi akhir untuk aplikasi.
    return inverse_label_map[prediksi_ensemble(index_pasien, BEST_W_FUZZY, BEST_W_CBF)] # Mengembalikan hasil prediksi akhir dalam bentuk label kategori diet.

def generate_rekomendasi_diet(pasien, kategori):                     # Membuat fungsi untuk menghasilkan rekomendasi diet berdasarkan kategori.
    rekomendasi = []                                                 # Membuat list kosong untuk rekomendasi diet.
    pantangan = []                                                   # Membuat list kosong untuk pantangan makanan.
    menu = []                                                        # Membuat list kosong untuk menu harian.

    if kategori == 'Ketat':                                          # Mengecek apakah kategori diet adalah ketat.
        rekomendasi = ['Diet rendah kalori', 'Perbanyak sayur hijau', 'Konsumsi protein tanpa lemak'] # Menentukan rekomendasi diet ketat.
        menu = ['Oatmeal + telur rebus', 'Salad sayur + ayam panggang', 'Sup sayur'] # Menentukan menu diet ketat.
        pantangan = ['Gorengan', 'Fast food', 'Minuman manis']       # Menentukan pantangan diet ketat.
    elif kategori == 'Sedang':                                       # Mengecek apakah kategori diet adalah sedang.
        rekomendasi = ['Diet seimbang', 'Kontrol porsi makan', 'Olahraga rutin'] # Menentukan rekomendasi diet sedang.
        menu = ['Nasi merah + ayam', 'Ikan bakar + sayur', 'Buah segar'] # Menentukan menu diet sedang.
        pantangan = ['Makanan tinggi gula', 'Lemak berlebih']        # Menentukan pantangan diet sedang.
    else:                                                            # Kondisi jika kategori diet adalah ringan.
        rekomendasi = ['Pola makan sehat', 'Pertahankan berat badan'] # Menentukan rekomendasi diet ringan.
        menu = ['Nasi + lauk + sayur', 'Buah harian']                # Menentukan menu diet ringan.

    kondisi_str = str(pasien['Kondisi Medis']).lower()               # Mengubah kondisi medis pasien menjadi huruf kecil.
    if 'diabetes' in kondisi_str:                                    # Mengecek apakah pasien memiliki diabetes.
        pantangan.append('Gula tinggi')                              # Menambahkan pantangan gula tinggi.
        menu.append('Makanan rendah indeks glikemik')                # Menambahkan menu rendah indeks glikemik.
    if 'hipertensi' in kondisi_str:                                  # Mengecek apakah pasien memiliki hipertensi.
        pantangan.append('Garam tinggi')                             # Menambahkan pantangan garam tinggi.
        menu.append('Makanan rendah sodium')                         # Menambahkan menu rendah sodium.
    if 'kolesterol' in kondisi_str:                                  # Mengecek apakah pasien memiliki kolesterol tinggi.
        pantangan.append('Lemak jenuh')                              # Menambahkan pantangan lemak jenuh.
        menu.append('Ikan omega-3')                                  # Menambahkan menu ikan omega-3.

    alergi_str = str(pasien['Alergi Makanan']).lower()               # Mengubah data alergi makanan menjadi huruf kecil.
    if alergi_str != 'nan' and alergi_str != '' and alergi_str != 'tidak ada': # Mengecek apakah pasien memiliki alergi makanan.
        pantangan.append('Hindari: ' + alergi_str)                   # Menambahkan alergi makanan ke daftar pantangan.

    return {                                                         # Mengembalikan hasil rekomendasi diet dalam bentuk dictionary.
        'Rekomendasi': list(set(rekomendasi)),                       # Mengembalikan rekomendasi tanpa duplikasi.
        'Menu Harian': list(set(menu)),                              # Mengembalikan menu harian tanpa duplikasi.
        'Pantangan': list(set(pantangan))                            # Mengembalikan pantangan tanpa duplikasi.
    }

def preprocess_new_data(new_patient_data, df_existing, scaler_obj, fitur_final_training_data): # Membuat fungsi preprocessing data baru untuk metode CBF.

    temp_df_single = pd.DataFrame([new_patient_data])                                      # Mengubah data pasien baru menjadi DataFrame sementara.

    required_original_cols = ['IMT','Usia','Lingkar Perut (cm)',                           # Mendefinisikan daftar kolom yang wajib ada.
                              'Kadar Lemak Tubuh (%)','Kadar Kolesterol (mg/dL)',
                              'Aktivitas_enc', 'Berat Badan (kg)', 'Tinggi Badan (m)',
                              'Denyut Jantung (bpm)', 'Kebutuhan Kalori (kkal)',
                              'Kondisi Medis', 'Alergi Makanan', 'Tekanan Darah']          # Menambahkan kolom lain yang digunakan fuzzy.

    for col in required_original_cols:                                                     # Melakukan perulangan untuk setiap kolom wajib.
        if col not in temp_df_single.columns:                                              # Mengecek apakah kolom belum tersedia.
            temp_df_single[col] = '' if isinstance(new_patient_data.get(col), str) else 0 # Mengisi kolom kosong dengan string kosong atau angka 0.

    original_fitur_numerik_cols = ['IMT','Usia','Lingkar Perut (cm)',                      # Mendefinisikan daftar fitur numerik asli.
                                   'Kadar Lemak Tubuh (%)','Kadar Kolesterol (mg/dL)',
                                   'Aktivitas_enc', 'Berat Badan (kg)', 'Tinggi Badan (m)',
                                   'Denyut Jantung (bpm)', 'Kebutuhan Kalori (kkal)']

    all_one_hot_cols = [col for col in fitur_final_training_data.columns if col not in original_fitur_numerik_cols] # Mengambil semua kolom hasil one-hot encoding.
    # Dynamically get condition and allergy columns from the training data
    existing_kondisi_cols = [col for col in df_existing['Kondisi Medis'].fillna('').str.get_dummies(sep=',').columns if col in all_one_hot_cols]
    existing_alergi_cols = [col for col in df_existing['Alergi Makanan'].fillna('').str.get_dummies(sep=',').columns if col in all_one_hot_cols]

    df_kondisi_new = temp_df_single['Kondisi Medis'].fillna('').str.get_dummies(sep=',')   # Mengubah kondisi medis pasien baru menjadi format one-hot encoding.
    df_alergi_new = temp_df_single['Alergi Makanan'].fillna('').str.get_dummies(sep=',')   # Mengubah alergi makanan pasien baru menjadi format one-hot encoding.

    # Ensure new one-hot columns match the existing ones
    df_kondisi_new = df_kondisi_new.reindex(columns=existing_kondisi_cols, fill_value=0)        # Menyesuaikan kolom kondisi medis dengan data training.
    df_alergi_new = df_alergi_new.reindex(columns=existing_alergi_cols, fill_value=0)           # Menyesuaikan kolom alergi makanan dengan data training.

    fitur_numerik_new = temp_df_single[original_fitur_numerik_cols].copy() # Mengambil fitur numerik dari pasien baru.

    fitur_numerik_new.fillna(df_existing[original_fitur_numerik_cols].mean(), inplace=True)             # Mengisi nilai kosong dengan rata-rata data training.
    fitur_final_new = pd.concat([fitur_numerik_new, df_kondisi_new, df_alergi_new], axis=1) # Menggabungkan fitur numerik, kondisi medis, dan alergi.
    fitur_final_new.fillna(0, inplace=True)                                                  # Mengisi nilai kosong dengan angka 0.

    fitur_final_new = fitur_final_new.reindex(columns=fitur_final_training_data.columns, fill_value=0) # Menyesuaikan urutan kolom dengan data training.

    scaled_new_data = scaler_obj.transform(fitur_final_new)                                  # Melakukan normalisasi data baru menggunakan scaler yang sudah dilatih.
    return scaled_new_data                                                                   # Mengembalikan data baru yang sudah diproses dan dinormalisasi.


# --- 4. Streamlit UI ---
st.title('🩺 Sistem Rekomendasi Diet Pasien Obesitas')                  # Menampilkan judul utama aplikasi Streamlit.
st.write('Aplikasi ini memberikan rekomendasi diet berdasarkan data pasien yang mirip dan logika fuzzy.') # Menampilkan deskripsi singkat aplikasi.
diet_info = {                                                           # Membuat dictionary awal untuk menyimpan hasil rekomendasi diet.
    'Rekomendasi': [],                                                  # Menyimpan daftar rekomendasi diet.
    'Menu Harian': [],                                                  # Menyimpan daftar menu harian.
    'Pantangan': []                                                     # Menyimpan daftar pantangan makanan.
}

final_prediction = None                                                 # Menginisialisasi variabel hasil prediksi akhir dengan nilai kosong.
selection_mode = st.radio(                                              # Membuat pilihan mode input menggunakan radio button.
    "Pilih mode input data:",                                           # Menampilkan label pilihan mode input.
    ('Pilih Pasien yang Ada', 'Masukkan Data Pasien Baru')              # Menyediakan dua opsi input data.
)

user_data = {}                                                          # Membuat dictionary kosong untuk menyimpan data pasien.
patient_index = -1                                                      # Menginisialisasi index pasien dengan nilai default -1.

if selection_mode == 'Pilih Pasien yang Ada':                           # Mengecek apakah pengguna memilih pasien yang sudah ada.
    st.subheader('Pilih Pasien dari Dataset')                           # Menampilkan subjudul untuk pemilihan pasien.
    patient_names = [''] + df['Nama Lengkap'].tolist()                  # Mengambil seluruh nama pasien dari dataset dan menambahkan opsi kosong.
    selected_name = st.selectbox('Pilih Nama Pasien', patient_names)    # Membuat dropdown untuk memilih nama pasien.

    if selected_name:                                                   # Mengecek apakah pengguna sudah memilih pasien.
        patient_index = df[df['Nama Lengkap'] == selected_name].index[0] # Mengambil index pasien berdasarkan nama yang dipilih.
        user_data = df.iloc[patient_index].to_dict()                    # Mengubah data pasien terpilih menjadi dictionary.

        aktivitas_map = {'rendah': 1, 'sedang': 2, 'tinggi': 3}         # Membuat mapping aktivitas fisik ke bentuk numerik.
        user_data['Aktivitas_enc'] = aktivitas_map.get(                 # Mengubah aktivitas fisik pasien menjadi nilai numerik.
            str(user_data.get('Aktivitas Fisik', 'sedang')).lower(), 2  # Mengambil nilai aktivitas fisik dan memberi default sedang.
        )

        st.write("**Data Pasien Terpilih:**")                           # Menampilkan teks informasi data pasien terpilih.
        display_df = pd.DataFrame([user_data]).drop(                    # Membuat DataFrame untuk ditampilkan ke aplikasi.
            columns=['Label_enc', 'TB_m'], errors='ignore'              # Menghapus kolom yang tidak perlu ditampilkan.
        )
        st.dataframe(display_df)


else: # Masukkan Data Pasien Baru                                      # Kondisi jika pengguna memilih memasukkan data pasien baru.
    st.subheader('Masukkan Data Pasien Secara Manual')                 # Menampilkan subjudul form input pasien baru.
    st.info('Untuk input data baru, rekomendasi akan menggunakan kombinasi Fuzzy Logic dan Content-Based Filtering. Namun, kesamaan CBF akan dihitung terhadap dataset yang sudah ada.') # Menampilkan informasi cara kerja sistem rekomendasi.
    col1, col2 = st.columns(2)                                         # Membagi tampilan input menjadi 2 kolom.

    with col1:                                                         # Membuat input pada kolom pertama.
        user_data['Nama Lengkap'] = st.text_input('Nama Lengkap', 'Pasien Baru') # Input nama lengkap pasien.
        user_data['Usia'] = st.number_input('Usia (tahun)', min_value=1, max_value=120, value=30) # Input usia pasien.
        user_data['Berat Badan (kg)'] = st.number_input('Berat Badan (kg)', min_value=10, max_value=300, value=70) # Input berat badan pasien.
        user_data['Tinggi Badan (m)'] = st.number_input('Tinggi Badan (m)', min_value=0.5, max_value=2.5, value=1.70, format="%.2f") # Input tinggi badan pasien.
        user_data['Lingkar Perut (cm)'] = st.number_input('Lingkar Perut (cm)', min_value=50, max_value=200, value=80) # Input lingkar perut pasien.
        user_data['Kadar Lemak Tubuh (%)'] = st.number_input('Kadar Lemak Tubuh (%)', min_value=5.0, max_value=60.0, value=25.0, format="%.1f") # Input kadar lemak tubuh pasien.
    with col2:                                                         # Membuat input pada kolom kedua.
        user_data['Denyut Jantung (bpm)'] = st.number_input('Denyut Jantung (bpm)', min_value=40, max_value=200, value=70) # Input denyut jantung pasien.
        user_data['Tekanan Darah'] = st.text_input('Tekanan Darah (Sistolik/Diastolik)', '120/80') # Input tekanan darah pasien.
        user_data['Kadar Kolesterol (mg/dL)'] = st.number_input('Kadar Kolesterol (mg/dL)', min_value=100.0, max_value=400.0, value=180.0, format="%.1f") # Input kadar kolesterol pasien.
        user_data['Aktivitas Fisik'] = st.selectbox('Aktivitas Fisik', ['Rendah', 'Sedang', 'Tinggi']) # Input tingkat aktivitas fisik pasien.
        user_data['Kondisi Medis'] = st.text_input('Kondisi Medis (pisahkan dengan koma)', 'Tidak ada') # Input kondisi medis pasien.
        user_data['Alergi Makanan'] = st.text_input('Alergi Makanan (pisahkan dengan koma)', 'Tidak ada') # Input alergi makanan pasien.
        user_data['Kebutuhan Kalori (kkal)'] = st.number_input('Kebutuhan Kalori (kkal)', min_value=1000, max_value=4000, value=2000) # Input kebutuhan kalori pasien.

    # Calculate IMT and encode Aktivitas Fisik for new data
    if user_data['Tinggi Badan (m)'] > 0:                              # Mengecek apakah tinggi badan valid.
        user_data['IMT'] = user_data['Berat Badan (kg)'] / (user_data['Tinggi Badan (m)'] ** 2) # Menghitung nilai IMT pasien.
    else:                                                              # Kondisi jika tinggi badan tidak valid.
        user_data['IMT'] = 0                                           # Mengatur nilai IMT menjadi 0.

    aktivitas_map_str = {'Rendah': 1, 'Sedang': 2, 'Tinggi': 3}        # Membuat mapping aktivitas fisik ke angka.
    user_data['Aktivitas_enc'] = aktivitas_map_str.get(user_data['Aktivitas Fisik'], 2) # Mengubah aktivitas fisik menjadi nilai numerik.
    if st.button('Hitung Rekomendasi untuk Data Baru'):                # Mengecek apakah tombol hitung ditekan.

        # Create a new row for calculation, including IMT for fuzzy and full features for CBF
        # Ensure all required columns are present in user_data, even if empty/default
        # This step is crucial for `preprocess_new_data`
        required_cols_for_df_conversion = [ 'ID Pasien', 'Nama Lengkap', 'Usia', 'Jenis Kelamin', 'Berat Badan (kg)', 'Tinggi Badan (m)', 'IMT', 'Lingkar Perut (cm)', 'Kadar Lemak Tubuh (%)', 'Denyut Jantung (bpm)', 'Tekanan Darah', 'Kadar Kolesterol (mg/dL)', 'Aktivitas Fisik', 'Kondisi Medis', 'Alergi Makanan', 'Kebutuhan Kalori (kkal)', 'Aktivitas_enc']
        for col in required_cols_for_df_conversion:
            if col not in user_data:
                if col in ['Nama Lengkap', 'Jenis Kelamin', 'Aktivitas Fisik', 'Kondisi Medis', 'Alergi Makanan', 'Tekanan Darah']:
                    user_data[col] = ''
                else:
                    user_data[col] = 0


        # --- Fuzzy Prediction for New Data ---
        fuzzy_score_new = prediksi_fuzzy(user_data)                    # Menghitung skor fuzzy pasien baru.

        fuzzy_label_new = kategori_diet(fuzzy_score_new)               # Mengubah skor fuzzy menjadi kategori diet.

        # --- CBF Prediction for New Data ---
        # Preprocess the new data using the saved scaler and original df structure
        processed_new_data_scaled = preprocess_new_data(user_data, df, scaler, fitur_final_training) # Melakukan preprocessing data baru untuk CBF.

        # Calculate similarity between new data and all existing data
        new_patient_similarity = cosine_similarity(                    # Menghitung tingkat kemiripan pasien baru dengan data training.
            processed_new_data_scaled,
            scaler.transform(fitur_final_training)
        )[0]

        sim_scores_new_patient = list(enumerate(new_patient_similarity)) # Membuat daftar index dan skor similarity.
        sim_scores_new_patient = sorted(                               # Mengurutkan similarity dari terbesar ke terkecil.
            sim_scores_new_patient, 
            key=lambda x: x[1],
            reverse=True
        )

        # Get labels from top_n similar patients in the existing dataset
        top_n_similar_indices = [idx for idx, _ in sim_scores_new_patient[1:6]] # Mengambil 5 pasien paling mirip (excluding itself at index 0).
        if top_n_similar_indices:                                      # Mengecek apakah ada pasien mirip.
            cbf_labels_new = [df.iloc[idx]['Label Diet'] for idx in top_n_similar_indices] # Mengambil label diet pasien mirip.
            vote_count_new = Counter(cbf_labels_new)                   # Menghitung jumlah kemunculan label diet.
            cbf_label_new = vote_count_new.most_common(1)[0][0]        # Mengambil label diet dengan voting terbanyak.
        else:                                                          # Kondisi jika tidak ada pasien mirip.
            cbf_label_new = 'Sedang'                                   # Menggunakan label default sedang.

        # --- Ensemble for New Data ---
        fuzzy_enc_new = label_map[fuzzy_label_new]                     # Mengubah label fuzzy menjadi numerik.
        cbf_enc_new = label_map[cbf_label_new]                         # Mengubah label CBF menjadi numerik.
        ensemble_score_new = BEST_W_FUZZY * fuzzy_enc_new + BEST_W_CBF * cbf_enc_new # Menghitung skor ensemble.
        if ensemble_score_new < 0.8:                                   # Mengecek apakah skor ensemble rendah.
            final_prediction = inverse_label_map[0]                    # Mengatur hasil prediksi menjadi diet ringan.
        elif ensemble_score_new < 1.8:                                 # Mengecek apakah skor ensemble sedang.
            final_prediction = inverse_label_map[1]                    # Mengatur hasil prediksi menjadi diet sedang.
        else:                                                          # Kondisi jika skor ensemble tinggi.
            final_prediction = inverse_label_map[2]                    # Mengatur hasil prediksi menjadi diet ketat.
        st.session_state['final_prediction_new_data'] = final_prediction # Menyimpan hasil prediksi ke session state.
        st.session_state['user_data_new_data'] = user_data             # Menyimpan data pasien baru ke session state.


# Display results logic
if (selection_mode == 'Pilih Pasien yang Ada' and patient_index != -1): # Mengecek apakah pengguna memilih pasien yang sudah ada dan index valid.
    final_prediction = prediksi_final_app(patient_index)               # Menghasilkan prediksi akhir kategori diet pasien.
    diet_info = generate_rekomendasi_diet(                             # Membuat rekomendasi diet berdasarkan data pasien.
        df.iloc[patient_index],
        final_prediction
    )
    st.subheader('Hasil Rekomendasi Diet')                             # Menampilkan subjudul hasil rekomendasi diet.
    st.success(f"Kategori Diet yang Direkomendasikan: **{final_prediction}**") # Menampilkan kategori diet hasil prediksi.
    st.write(f"IMT Pasien: **{df.iloc[patient_index]['IMT']:.2f}**")   # Menampilkan nilai IMT pasien.

elif (selection_mode == 'Masukkan Data Pasien Baru' and 'final_prediction_new_data' in st.session_state): # Mengecek apakah pengguna memasukkan data baru dan hasil prediksi tersedia.
    final_prediction = st.session_state['final_prediction_new_data']   # Mengambil hasil prediksi dari session state.
    user_data = st.session_state['user_data_new_data']                 # Mengambil data pasien baru dari session state.
    diet_info = generate_rekomendasi_diet(                             # Membuat rekomendasi diet untuk pasien baru.
        user_data,
        final_prediction
    )
    st.subheader('Hasil Rekomendasi Diet')                             # Menampilkan subjudul hasil rekomendasi diet.
    st.success(f"Kategori Diet yang Direkomendasikan: **{final_prediction}**") # Menampilkan kategori diet hasil prediksi.
    if 'IMT' in user_data:                                             # Mengecek apakah nilai IMT tersedia.
        st.write(f"IMT Pasien: **{user_data['IMT']:.2f}**")            # Menampilkan nilai IMT pasien baru.
    else:                                                              # Kondisi jika IMT tidak tersedia.
        st.write("IMT Pasien: Belum terhitung atau tidak valid")       # Menampilkan pesan jika IMT belum valid.

if final_prediction is not None:                                       # Mengecek apakah hasil prediksi tersedia.
    st.write("**Rekomendasi Utama:**")                                 # Menampilkan judul rekomendasi utama.
    for rec in diet_info['Rekomendasi']:                               # Melakukan perulangan pada daftar rekomendasi.
        st.write(f"- {rec}")                                           # Menampilkan setiap rekomendasi diet.

    st.write("**Menu Harian yang Disarankan:**")                       # Menampilkan judul menu harian.
    for menu_item in diet_info['Menu Harian']:                         # Melakukan perulangan pada daftar menu harian.
        st.write(f"- {menu_item}")                                     # Menampilkan setiap menu harian.

    st.write("**Pantangan Makanan:**")                                 # Menampilkan judul pantangan makanan.
    if diet_info['Pantangan']:                                         # Mengecek apakah ada pantangan makanan.
        for pantang in diet_info['Pantangan']:                         # Melakukan perulangan pada daftar pantangan.
            st.write(f"- {pantang}")                                   # Menampilkan setiap pantangan makanan.
    else:                                                              # Kondisi jika tidak ada pantangan tambahan.
        st.write("- Tidak ada pantangan spesifik tambahan.")           # Menampilkan pesan bahwa tidak ada pantangan tambahan.
