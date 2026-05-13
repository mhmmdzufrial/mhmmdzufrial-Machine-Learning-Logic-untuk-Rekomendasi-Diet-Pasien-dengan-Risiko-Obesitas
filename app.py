
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter

# --- 1. Load Model Components and Data ---
@st.cache_resource
def load_model_components():
    df = pd.read_pickle('processed_df.pkl')
    similarity_matrix = np.load('similarity_matrix.npy')
    with open('model_params.pkl', 'rb') as f:
        model_params = pickle.load(f)

    # Reconstruct features for CBF within the app, using the same logic as the notebook
    df_kondisi = df['Kondisi Medis'].fillna('').str.get_dummies(sep=',')
    df_alergi = df['Alergi Makanan'].fillna('').str.get_dummies(sep=',')
    fitur_numerik = df[['IMT','Usia','Lingkar Perut (cm)',
                        'Kadar Lemak Tubuh (%)','Kadar Kolesterol (mg/dL)',
                        'Aktivitas_enc', 'Berat Badan (kg)', 'Tinggi Badan (m)',
                        'Denyut Jantung (bpm)', 'Kebutuhan Kalori (kkal)']]
    fitur_final_for_scaling = pd.concat([fitur_numerik, df_kondisi, df_alergi], axis=1)
    fitur_final_for_scaling.fillna(0, inplace=True)

    # The scaler needs to be fitted on the same data it was trained on.
    # For the Streamlit app, we will assume the min/max values are from the training data.
    # In a real-world scenario, you would save the fitted scaler object.
    # For simplicity, we are re-creating it here and assuming the training data's min/max bounds.
    # A more robust solution would involve saving and loading the `scaler` object itself.
    scaler = MinMaxScaler()
    scaler.fit(fitur_final_for_scaling) # Fit the scaler to the loaded data

    return df, similarity_matrix, model_params, scaler # Pass scaler for new data transformation

df, similarity_matrix, model_params, scaler = load_model_components()
THRESHOLD_1 = model_params['THRESHOLD_1']
THRESHOLD_2 = model_params['THRESHOLD_2']
BEST_W_FUZZY = model_params['BEST_W_FUZZY']
BEST_W_CBF = model_params['BEST_W_CBF']
label_map = model_params['label_map']
inverse_label_map = model_params['inverse_label_map']

# --- 2. Fuzzy Logic System (Re-definition for Streamlit) ---
# Define Fuzzy variables
imt = ctrl.Antecedent(np.arange(10, 55, 1), 'imt')
lingkar = ctrl.Antecedent(np.arange(60, 140, 1), 'lingkar')
lemak = ctrl.Antecedent(np.arange(5, 45, 1), 'lemak')
aktivitas = ctrl.Antecedent(np.arange(1, 4, 0.5), 'aktivitas')
tekanan = ctrl.Antecedent(np.arange(80, 200, 1), 'tekanan')
kolesterol = ctrl.Antecedent(np.arange(100, 310, 1), 'kolesterol')
kondisi = ctrl.Antecedent(np.arange(0, 11, 1), 'kondisi')
diet = ctrl.Consequent(np.arange(0, 101, 1), 'diet')

# Membership functions
imt['rendah']    = fuzz.trimf(imt.universe, [10, 18, 24])
imt['normal']    = fuzz.trimf(imt.universe, [22, 25, 28])
imt['tinggi']    = fuzz.trimf(imt.universe, [27, 33, 40])
imt['obesitas']  = fuzz.trimf(imt.universe, [35, 45, 55])

lingkar['normal'] = fuzz.trimf(lingkar.universe, [60, 75, 90])
lingkar['tinggi'] = fuzz.trimf(lingkar.universe, [85, 105, 140])

lemak['rendah'] = fuzz.trimf(lemak.universe, [5, 12, 20])
lemak['normal'] = fuzz.trimf(lemak.universe, [18, 24, 30])
lemak['tinggi'] = fuzz.trimf(lemak.universe, [28, 35, 45])

aktivitas['rendah'] = fuzz.trimf(aktivitas.universe, [1, 1, 1.5])
aktivitas['sedang'] = fuzz.trimf(aktivitas.universe, [1.5, 2, 2.5])
aktivitas['tinggi'] = fuzz.trimf(aktivitas.universe, [2, 3, 3])

tekanan['normal'] = fuzz.trimf(tekanan.universe, [80, 110, 130])
tekanan['pra_tinggi'] = fuzz.trimf(tekanan.universe, [125, 140, 150])
tekanan['tinggi'] = fuzz.trimf(tekanan.universe, [145, 170, 200])

kolesterol['normal'] = fuzz.trimf(kolesterol.universe, [100, 160, 200])
kolesterol['tinggi'] = fuzz.trimf(kolesterol.universe, [190, 250, 310])

kondisi['ringan'] = fuzz.trimf(kondisi.universe, [0, 1, 2])
kondisi['sedang'] = fuzz.trimf(kondisi.universe, [1, 3, 5])
kondisi['berat'] = fuzz.trimf(kondisi.universe, [4, 7, 10])

diet['ringan'] = fuzz.trimf(diet.universe, [0, 20, 40])
diet['sedang'] = fuzz.trimf(diet.universe, [30, 50, 70])
diet['ketat'] = fuzz.trimf(diet.universe, [60, 80, 100])

# Fuzzy Rules
rules = []
rules.append(ctrl.Rule(imt['rendah'] & aktivitas['tinggi'], diet['ringan']))
rules.append(ctrl.Rule(imt['normal'] & aktivitas['tinggi'] & tekanan['normal'], diet['ringan']))
rules.append(ctrl.Rule(imt['normal'] & kolesterol['normal'] & kondisi['ringan'], diet['ringan']))
rules.append(ctrl.Rule(imt['rendah'] & lemak['rendah'], diet['ringan']))
rules.append(ctrl.Rule(imt['normal'], diet['sedang']))
rules.append(ctrl.Rule(imt['tinggi'] & aktivitas['tinggi'], diet['sedang']))
rules.append(ctrl.Rule(imt['tinggi'] & lemak['normal'], diet['sedang']))
rules.append(ctrl.Rule(imt['obesitas'] & aktivitas['tinggi'] & tekanan['normal'], diet['sedang']))
rules.append(ctrl.Rule(imt['normal'] & kolesterol['tinggi'], diet['sedang']))
rules.append(ctrl.Rule(aktivitas['sedang'] & tekanan['pra_tinggi'], diet['sedang']))
rules.append(ctrl.Rule(imt['obesitas'], diet['ketat']))
rules.append(ctrl.Rule(imt['tinggi'] & aktivitas['rendah'], diet['ketat']))
rules.append(ctrl.Rule(kolesterol['tinggi'] & lemak['tinggi'], diet['ketat']))
rules.append(ctrl.Rule(tekanan['tinggi'] & kondisi['berat'], diet['ketat']))
rules.append(ctrl.Rule(imt['obesitas'] & tekanan['tinggi'], diet['ketat']))
rules.append(ctrl.Rule(kondisi['berat'] & lemak['tinggi'], diet['ketat']))
rules.append(ctrl.Rule(lingkar['tinggi'] & imt['obesitas'], diet['ketat']))
rules.append(ctrl.Rule(imt['obesitas'] & kolesterol['tinggi'], diet['ketat']))
rules.append(ctrl.Rule(imt['normal'] | imt['tinggi'] | imt['obesitas'], diet['sedang'])) # Default rule

diet_ctrl = ctrl.ControlSystem(rules)
diet_sim = ctrl.ControlSystemSimulation(diet_ctrl)

# --- 3. Prediction Functions (from notebook) ---
def hitung_kondisi_medis(kondisi_str):
    if not isinstance(kondisi_str, str): # Handle NaN or non-string inputs
        return 0
    kondisi_list = kondisi_str.split(',')
    kondisi_list = [k.strip() for k in kondisi_list if k.strip()]
    return min(len(kondisi_list), 10)

def extract_systolic(tekanan_str):
    match = re.match(r'(\\d+)/', str(tekanan_str))
    if match:
        return int(match.group(1))
    return 120 # Default if format is invalid

def prediksi_fuzzy(input_data):
    try:
        diet_sim.input['imt'] = input_data['IMT']
        diet_sim.input['lingkar'] = input_data['Lingkar Perut (cm)']
        diet_sim.input['lemak'] = input_data['Kadar Lemak Tubuh (%)']
        diet_sim.input['aktivitas'] = input_data['Aktivitas_enc']
        diet_sim.input['tekanan'] = extract_systolic(input_data['Tekanan Darah'])
        diet_sim.input['kolesterol'] = input_data['Kadar Kolesterol (mg/dL)']
        diet_sim.input['kondisi'] = hitung_kondisi_medis(input_data['Kondisi Medis'])
        diet_sim.compute()
        return diet_sim.output['diet']
    except Exception as e:
        # Fallback to IMT-based default if fuzzy computation fails
        if input_data['IMT'] > 30:
            return 80.0
        elif input_data['IMT'] > 25:
            return 50.0
        else:
            return 20.0

def kategori_diet(score):
    if score < THRESHOLD_1:
        return 'Ringan'
    elif score < THRESHOLD_2:
        return 'Sedang'
    else:
        return 'Ketat'

def rekomendasi_cbf(index_pasien, top_n=5):
    sim_scores = list(enumerate(similarity_matrix[index_pasien]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    hasil = []
    for i, score in sim_scores:
        hasil.append({
            'Index': i,
            'Nama': df.iloc[i]['Nama Lengkap'],
            'Similarity': round(score, 4),
            'Label': df.iloc[i]['Label Diet']
        })
    return hasil

def get_cbf_majority_vote(index_pasien, top_n=5):
    similar = rekomendasi_cbf(index_pasien, top_n)
    labels = [s['Label'] for s in similar]
    if not labels: # Handle case where no similar patients are found
        return 'Ringan' # Default label
    vote_count = Counter(labels)
    return vote_count.most_common(1)[0][0]

def prediksi_ensemble(index_pasien, w_fuzzy, w_cbf):
    row = df.iloc[index_pasien]
    fuzzy_score = prediksi_fuzzy(row)
    fuzzy_label = kategori_diet(fuzzy_score)
    fuzzy_enc = label_map[fuzzy_label]
    cbf_label = get_cbf_majority_vote(index_pasien, top_n=5)
    cbf_enc = label_map[cbf_label]

    ensemble_score = w_fuzzy * fuzzy_enc + w_cbf * cbf_enc
    if ensemble_score < 0.8:
        return 0
    elif ensemble_score < 1.8:
        return 1
    else:
        return 2

def prediksi_final_app(index_pasien):
    return inverse_label_map[prediksi_ensemble(index_pasien, BEST_W_FUZZY, BEST_W_CBF)]

def generate_rekomendasi_diet(pasien, kategori):
    rekomendasi = []
    pantangan = []
    menu = []

    if kategori == 'Ketat':
        rekomendasi = ['Diet rendah kalori', 'Perbanyak sayur hijau', 'Konsumsi protein tanpa lemak']
        menu = ['Oatmeal + telur rebus', 'Salad sayur + ayam panggang', 'Sup sayur']
        pantangan = ['Gorengan', 'Fast food', 'Minuman manis']

    elif kategori == 'Sedang':
        rekomendasi = ['Diet seimbang', 'Kontrol porsi makan', 'Olahraga rutin']
        menu = ['Nasi merah + ayam', 'Ikan bakar + sayur', 'Buah segar']
        pantangan = ['Makanan tinggi gula', 'Lemak berlebih']

    else:
        rekomendasi = ['Pola makan sehat', 'Pertahankan berat badan']
        menu = ['Nasi + lauk + sayur', 'Buah harian']

    kondisi_str = str(pasien['Kondisi Medis']).lower()
    if 'diabetes' in kondisi_str:
        pantangan.append('Gula tinggi')
        menu.append('Makanan rendah indeks glikemik')
    if 'hipertensi' in kondisi_str:
        pantangan.append('Garam tinggi')
        menu.append('Makanan rendah sodium')
    if 'kolesterol' in kondisi_str:
        pantangan.append('Lemak jenuh')
        menu.append('Ikan omega-3')

    alergi_str = str(pasien['Alergi Makanan']).lower()
    if alergi_str != 'nan' and alergi_str != '':
        pantangan.append('Hindari: ' + alergi_str)

    return {
        'Rekomendasi': list(set(rekomendasi)),
        'Menu Harian': list(set(menu)),
        'Pantangan': list(set(pantangan))
    }

# --- 4. Streamlit UI ---
st.title('🩺 Sistem Rekomendasi Diet Pasien Obesitas')
st.write('Aplikasi ini memberikan rekomendasi diet berdasarkan data pasien yang mirip dan logika fuzzy.')

# Option to select an existing patient or enter new data
selection_mode = st.radio(
    "Pilih mode input data:",
    ('Pilih Pasien yang Ada', 'Masukkan Data Pasien Baru')
)

user_data = {}
patient_index = -1

if selection_mode == 'Pilih Pasien yang Ada':
    st.subheader('Pilih Pasien dari Dataset')
    patient_names = [''] + df['Nama Lengkap'].tolist()
    selected_name = st.selectbox('Pilih Nama Pasien', patient_names)

    if selected_name:
        patient_index = df[df['Nama Lengkap'] == selected_name].index[0]
        user_data = df.iloc[patient_index].to_dict()
        # Override IMT if it was previously calculated (to avoid re-calculation errors from raw inputs)
        user_data['IMT'] = user_data['Berat Badan (kg)'] / (user_data['Tinggi Badan (m)'] ** 2)
        # Ensure Aktivitas_enc is present
        aktivitas_map = {'rendah': 1, 'sedang': 2, 'tinggi': 3}
        user_data['Aktivitas_enc'] = aktivitas_map.get(str(user_data['Aktivitas Fisik']).lower(), 2) # Default to sedang
        st.write("**Data Pasien Terpilih:**")
        st.dataframe(pd.DataFrame([user_data]))


else: # Masukkan Data Pasien Baru
    st.subheader('Masukkan Data Pasien Secara Manual')
    st.warning('Fitur input data baru saat ini hanya menggunakan Fuzzy Logic untuk prediksi, sehingga rekomendasi mungkin kurang akurat dibandingkan memilih pasien yang sudah ada dalam dataset.')

    col1, col2 = st.columns(2)
    with col1:
        user_data['Nama Lengkap'] = st.text_input('Nama Lengkap', 'Pasien Baru')
        user_data['Usia'] = st.number_input('Usia (tahun)', min_value=1, max_value=120, value=30)
        user_data['Berat Badan (kg)'] = st.number_input('Berat Badan (kg)', min_value=10, max_value=300, value=70)
        user_data['Tinggi Badan (m)'] = st.number_input('Tinggi Badan (m)', min_value=0.5, max_value=2.5, value=1.70, format="%.2f")
        user_data['Lingkar Perut (cm)'] = st.number_input('Lingkar Perut (cm)', min_value=50, max_value=200, value=80)
        user_data['Kadar Lemak Tubuh (%)'] = st.number_input('Kadar Lemak Tubuh (%)', min_value=5.0, max_value=60.0, value=25.0, format="%.1f")
    with col2:
        user_data['Denyut Jantung (bpm)'] = st.number_input('Denyut Jantung (bpm)', min_value=40, max_value=200, value=70)
        user_data['Tekanan Darah'] = st.text_input('Tekanan Darah (Sistolik/Diastolik)', '120/80')
        user_data['Kadar Kolesterol (mg/dL)'] = st.number_input('Kadar Kolesterol (mg/dL)', min_value=100.0, max_value=400.0, value=180.0, format="%.1f")
        user_data['Aktivitas Fisik'] = st.selectbox('Aktivitas Fisik', ['Rendah', 'Sedang', 'Tinggi'])
        user_data['Kondisi Medis'] = st.text_input('Kondisi Medis (pisahkan dengan koma)', 'Tidak ada')
        user_data['Alergi Makanan'] = st.text_input('Alergi Makanan (pisahkan dengan koma)', 'Tidak ada')

    # Calculate IMT and encode Aktivitas Fisik for new data
    if user_data['Tinggi Badan (m)'] > 0:
        user_data['IMT'] = user_data['Berat Badan (kg)'] / (user_data['Tinggi Badan (m)'] ** 2)
    else:
        user_data['IMT'] = 0
    aktivitas_map = {'Rendah': 1, 'Sedang': 2, 'Tinggi': 3}
    user_data['Aktivitas_enc'] = aktivitas_map.get(user_data['Aktivitas Fisik'], 2)

    if st.button('Hitung Rekomendasi'):
        # For new data, CBF part won't be accurate, but fuzzy can still run
        st.info('Untuk input data baru, rekomendasi hanya didasarkan pada Fuzzy Logic.')
        patient_index = -2 # Dummy index to indicate new patient


if (selection_mode == 'Pilih Pasien yang Ada' and selected_name) or (selection_mode == 'Masukkan Data Pasien Baru' and patient_index != -1):
    st.subheader('Hasil Rekomendasi Diet')

    if selection_mode == 'Pilih Pasien yang Ada':
        final_prediction = prediksi_final_app(patient_index)
        diet_info = generate_rekomendasi_diet(df.iloc[patient_index], final_prediction)
        st.success(f"Kategori Diet yang Direkomendasikan: **{final_prediction}**")

    elif selection_mode == 'Masukkan Data Pasien Baru' and patient_index == -2:
        # For new data, CBF part won't be accurate, but fuzzy can still run
        fuzzy_score_new = prediksi_fuzzy(user_data)
        fuzzy_label_new = kategori_diet(fuzzy_score_new)
        final_prediction = fuzzy_label_new
        diet_info = generate_rekomendasi_diet(user_data, final_prediction)
        st.success(f"Kategori Diet yang Direkomendasikan (Fuzzy Saja): **{final_prediction}**")

    if 'IMT' in user_data:
        st.write(f"IMT Pasien: **{user_data['IMT']:.2f}**")
    else:
        st.write("IMT Pasien: Belum terhitung atau tidak valid")

    st.write("**Rekomendasi Utama:**")
    for rec in diet_info['Rekomendasi']:
        st.write(f"- {rec}")

    st.write("**Menu Harian yang Disarankan:**")
    for menu_item in diet_info['Menu Harian']:
        st.write(f"- {menu_item}")

    st.write("**Pantangan Makanan:**")
    if diet_info['Pantangan']:
        for pantang in diet_info['Pantangan']:
            st.write(f"- {pantang}")
    else:
        st.write("- Tidak ada pantangan spesifik tambahan.")
