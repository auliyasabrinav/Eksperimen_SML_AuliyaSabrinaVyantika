import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_and_preprocess(df):
    data = df.dropna().copy()
    kolom_diubah = ['jenis_kelamin', 'status kredit']
    data[kolom_diubah] = data[kolom_diubah].astype('category')
    data["jenis_kelamin"] = data["jenis_kelamin"].replace({
        "WANITA": "P",
        "PEREMPUAN": "P",
        "LAKI-LAKI": "L",
        "PRIA": "L"
    })
    if 'nama_nasabah' in data.columns:
        data.drop('nama_nasabah', axis=1, inplace=True)
    label_encoder = LabelEncoder()
    data['jenis_kelamin'] = label_encoder.fit_transform(data['jenis_kelamin'])
    return data

def preprocess(filepath):
    df = load_data(filepath)
    processed_data = clean_and_preprocess(df)
    return processed_data

if __name__ == "__main__":
    raw_path = 'namadataset_raw/creditapproval-data_raw.csv'   # path dataset mentah
    processed_path = 'preprocessing/namadataset_preprocessing/creditapproval-data_processed.csv'  # path hasil preprocess

    os.makedirs('namadataset_preprocessing', exist_ok=True)

    data_ready = preprocess(raw_path)
    data_ready.to_csv(processed_path, index=False)
    print(f"Data berhasil diproses dan disimpan di {processed_path}")
