
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import re
from collections import Counter

# --- 1. Load Model Components and Data ---
@st.cache_resource
def load_model_components():
    df_loaded = pd.read_pickle('processed_df.pkl')
    similarity_matrix_loaded = np.load('similarity_matrix.npy')
    with open('model_params.pkl', 'rb') as f:
        model_params_loaded = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler_loaded = pickle.load(f)

    # Reconstruct fitur_final from the loaded df_loaded to be used for CBF against new data
    df_kondisi_loaded = df_loaded['Kondisi Medis'].fillna('').str.get_dummies(sep=',')
    df_alergi_loaded = df_loaded['Alergi Makanan'].fillna('').str.get_dummies(sep=',')
    fitur_numerik_loaded = df_loaded[['IMT','Usia','Lingkar Perut (cm)',
                                      'Kadar Lemak Tubuh (%)','Kadar Kolesterol (mg/dL)',
                                      'Aktivitas_enc', 'Berat Badan (kg)', 'Tinggi Badan (m)',
                                      'Denyut Jantung (bpm)', 'Kebutuhan Kalori (kkal)']]
    fitur_final_loaded = pd.concat([fitur_numerik_loaded, df_kondisi_loaded, df_alergi_loaded], axis=1)
    fitur_final_loaded.fillna(0, inplace=True)

    return df_loaded, similarity_matrix_loaded, model_params_loaded, scaler_loaded, fitur_final_loaded

df, similarity_matrix, model_params, scaler, fitur_final_training = load_model_components()
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
    if not isinstance(kondisi_str, str):
        return 0
    kondisi_list = kondisi_str.split(',')
    kondisi_list = [k.strip() for k in kondisi_list if k.strip()]
    return min(len(kondisi_list), 10)

def extract_systolic(tekanan_str):
    match = re.match(r'(\d+)/', str(tekanan_str))
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
    if not labels:
        return 'Sedang'
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
    if alergi_str != 'nan' and alergi_str != '' and alergi_str != 'tidak ada':
        pantangan.append('Hindari: ' + alergi_str)

    return {
        'Rekomendasi': list(set(rekomendasi)),
        'Menu Harian': list(set(menu)),
        'Pantangan': list(set(pantangan))
    }

# --- Helper function for new data preprocessing for CBF ---
def preprocess_new_data(new_patient_data, df_existing, scaler_obj, fitur_final_training_data):
    # Create a temporary DataFrame for the new patient, ensuring all expected columns
    temp_df_single = pd.DataFrame([new_patient_data])

    # Ensure all original columns used for feature extraction are present, fill missing with '' or 0
    # This part should mimic the original preprocessing in the notebook
    required_original_cols = ['IMT','Usia','Lingkar Perut (cm)',
                              'Kadar Lemak Tubuh (%)','Kadar Kolesterol (mg/dL)',
                              'Aktivitas_enc', 'Berat Badan (kg)', 'Tinggi Badan (m)',
                              'Denyut Jantung (bpm)', 'Kebutuhan Kalori (kkal)',
                              'Kondisi Medis', 'Alergi Makanan', 'Tekanan Darah'] # Add any other used in fuzzy

    for col in required_original_cols:
        if col not in temp_df_single.columns:
            temp_df_single[col] = '' if temp_df_single[col].dtype == 'object' else 0 # Default to empty string or 0

    # Extract conditions and allergies for new data
    # Get all unique conditions/allergies from the training data first
    # This can be derived from fitur_final_training_data columns that are not in fitur_numerik_loaded
    original_fitur_numerik_cols = ['IMT','Usia','Lingkar Perut (cm)',
                                   'Kadar Lemak Tubuh (%)','Kadar Kolesterol (mg/dL)',
                                   'Aktivitas_enc', 'Berat Badan (kg)', 'Tinggi Badan (m)',
                                   'Denyut Jantung (bpm)', 'Kebutuhan Kalori (kkal)']

    all_one_hot_cols = [col for col in fitur_final_training_data.columns if col not in original_fitur_numerik_cols]
    all_kondisi_cols = [col for col in all_one_hot_cols if col not in df_existing['Alergi Makanan'].fillna('').str.get_dummies(sep=',').columns]
    all_alergi_cols = [col for col in all_one_hot_cols if col not in df_existing['Kondisi Medis'].fillna('').str.get_dummies(sep=',').columns]


    df_kondisi_new = temp_df_single['Kondisi Medis'].fillna('').str.get_dummies(sep=',')
    df_alergi_new = temp_df_single['Alergi Makanan'].fillna('').str.get_dummies(sep=',')

    # Reindex to ensure all columns from training data are present, fill missing with 0
    df_kondisi_new = df_kondisi_new.reindex(columns=all_kondisi_cols, fill_value=0)
    df_alergi_new = df_alergi_new.reindex(columns=all_alergi_cols, fill_value=0)


    # Select numeric features for new data. Ensure 'Tekanan Darah' is handled if it impacts IMT, etc.
    fitur_numerik_new = temp_df_single[['IMT','Usia','Lingkar Perut (cm)',
                                        'Kadar Lemak Tubuh (%)','Kadar Kolesterol (mg/dL)',
                                        'Aktivitas_enc', 'Berat Badan (kg)', 'Tinggi Badan (m)',
                                        'Denyut Jantung (bpm)', 'Kebutuhan Kalori (kkal)']]
    fitur_numerik_new.fillna(df_existing.mean(numeric_only=True), inplace=True)

    fitur_final_new = pd.concat([fitur_numerik_new, df_kondisi_new, df_alergi_new], axis=1)
    fitur_final_new.fillna(0, inplace=True)

    # Align columns before scaling - crucial for consistent feature order
    fitur_final_new = fitur_final_new.reindex(columns=fitur_final_training_data.columns, fill_value=0)

    # Scale the new data using the *fitted* scaler
    scaled_new_data = scaler_obj.transform(fitur_final_new)
    return scaled_new_data



# --- 4. Streamlit UI ---
st.title('🩺 Sistem Rekomendasi Diet Pasien Obesitas')
st.write('Aplikasi ini memberikan rekomendasi diet berdasarkan data pasien yang mirip dan logika fuzzy.')

# Initialize diet_info to avoid NameError if no patient is selected/data entered initially
diet_info = {
    'Rekomendasi': [],
    'Menu Harian': [],
    'Pantangan': []
}
final_prediction = None # Initialize final_prediction

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
        # Ensure 'Aktivitas_enc' is correctly mapped as it might not be in original 'Aktivitas Fisik' string format
        aktivitas_map = {'rendah': 1, 'sedang': 2, 'tinggi': 3}
        user_data['Aktivitas_enc'] = aktivitas_map.get(str(user_data.get('Aktivitas Fisik', 'sedang')).lower(), 2)

        st.write("**Data Pasien Terpilih:**")
        display_df = pd.DataFrame([user_data]).drop(columns=['Label_enc', 'TB_m'], errors='ignore')
        st.dataframe(display_df)


else: # Masukkan Data Pasien Baru
    st.subheader('Masukkan Data Pasien Secara Manual')
    st.info('Untuk input data baru, rekomendasi akan menggunakan kombinasi Fuzzy Logic dan Content-Based Filtering. Namun, kesamaan CBF akan dihitung terhadap dataset yang sudah ada.')

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
        user_data['Kebutuhan Kalori (kkal)'] = st.number_input('Kebutuhan Kalori (kkal)', min_value=1000, max_value=4000, value=2000)

    # Calculate IMT and encode Aktivitas Fisik for new data
    if user_data['Tinggi Badan (m)'] > 0:
        user_data['IMT'] = user_data['Berat Badan (kg)'] / (user_data['Tinggi Badan (m)'] ** 2)
    else:
        user_data['IMT'] = 0
    aktivitas_map_str = {'Rendah': 1, 'Sedang': 2, 'Tinggi': 3}
    user_data['Aktivitas_enc'] = aktivitas_map_str.get(user_data['Aktivitas Fisik'], 2)

    if st.button('Hitung Rekomendasi untuk Data Baru'):
        # Create a new row for calculation, including IMT for fuzzy and full features for CBF
        new_patient_df = pd.DataFrame([user_data])

        # --- Fuzzy Prediction for New Data ---
        fuzzy_score_new = prediksi_fuzzy(user_data)
        fuzzy_label_new = kategori_diet(fuzzy_score_new)

        # --- CBF Prediction for New Data ---
        # Preprocess the new data using the saved scaler and original df structure
        processed_new_data_scaled = preprocess_new_data(user_data, df, scaler, fitur_final_training)

        # Calculate similarity between new data and all existing data
        new_patient_similarity = cosine_similarity(processed_new_data_scaled, scaler.transform(fitur_final_training))[0]

        # Find the most similar existing patient for CBF vote
        # Exclude self-similarity if the new patient were already in df, but here it's new
        sim_scores_new_patient = list(enumerate(new_patient_similarity))
        sim_scores_new_patient = sorted(sim_scores_new_patient, key=lambda x: x[1], reverse=True)

        # Get labels from top_n similar patients in the existing dataset
        top_n_similar_indices = [idx for idx, _ in sim_scores_new_patient[:5]] # Top 5 similar patients

        if top_n_similar_indices:
            cbf_labels_new = [df.iloc[idx]['Label Diet'] for idx in top_n_similar_indices]
            vote_count_new = Counter(cbf_labels_new)
            cbf_label_new = vote_count_new.most_common(1)[0][0]
        else:
            cbf_label_new = 'Sedang' # Fallback if no similar found (unlikely with this setup)

        # --- Ensemble for New Data ---
        fuzzy_enc_new = label_map[fuzzy_label_new]
        cbf_enc_new = label_map[cbf_label_new]
        ensemble_score_new = BEST_W_FUZZY * fuzzy_enc_new + BEST_W_CBF * cbf_enc_new

        if ensemble_score_new < 0.8:
            final_prediction = inverse_label_map[0]
        elif ensemble_score_new < 1.8:
            final_prediction = inverse_label_map[1]
        else:
            final_prediction = inverse_label_map[2]

        st.session_state['final_prediction_new_data'] = final_prediction
        st.session_state['user_data_new_data'] = user_data


# Display results logic
if (selection_mode == 'Pilih Pasien yang Ada' and patient_index != -1):
    final_prediction = prediksi_final_app(patient_index)
    diet_info = generate_rekomendasi_diet(df.iloc[patient_index], final_prediction)
    st.subheader('Hasil Rekomendasi Diet')
    st.success(f"Kategori Diet yang Direkomendasikan: **{final_prediction}**")
    st.write(f"IMT Pasien: **{df.iloc[patient_index]['IMT']:.2f}**")

elif (selection_mode == 'Masukkan Data Pasien Baru' and 'final_prediction_new_data' in st.session_state):
    final_prediction = st.session_state['final_prediction_new_data']
    user_data = st.session_state['user_data_new_data']
    diet_info = generate_rekomendasi_diet(user_data, final_prediction)
    st.subheader('Hasil Rekomendasi Diet')
    st.success(f"Kategori Diet yang Direkomendasikan: **{final_prediction}**")
    if 'IMT' in user_data:
        st.write(f"IMT Pasien: **{user_data['IMT']:.2f}**")
    else:
        st.write("IMT Pasien: Belum terhitung atau tidak valid")


if final_prediction is not None:
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
