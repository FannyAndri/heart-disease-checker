import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

file_path = r'C:\Users\fanny\Downloads\heart.csv'
data = pd.read_csv(file_path)

from imblearn.over_sampling import SMOTE

fitur_kategorikal = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for kolom in fitur_kategorikal:
    data[kolom] = LabelEncoder().fit_transform(data[kolom])

scaler = MinMaxScaler()
fitur_numerik = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
data[fitur_numerik] = scaler.fit_transform(data[fitur_numerik])

heart_disease_counts = data['HeartDisease'].value_counts()
total_samples = len(data)
heart_disease_percentages = (heart_disease_counts / total_samples) * 100

print("Distribusi HeartDisease di dataset:")
print(heart_disease_counts)
print(f"\nPersentase:")
for label, percentage in heart_disease_percentages.items():
    print(f"HeartDisease = {label}: {percentage:.2f}%")

X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def tampilkan_informasi_data(data):
    print("\nInformasi Dataset:")
    print(f"Jumlah Baris: {data.shape[0]}")
    print(f"Jumlah Kolom: {data.shape[1]}")
    print("\nLima Baris Pertama Dataset:")
    print(data.head())
    print("\nLima Baris Terakhir Dataset:")
    print(data.tail())

model_knn = KNeighborsClassifier(n_neighbors=3)
model_nb = GaussianNB()
model_dt = DecisionTreeClassifier(random_state=42)

stacking_ensemble = StackingClassifier(
    estimators=[
        ('knn', model_knn),
        ('naive_bayes', model_nb),
        ('decision_tree', model_dt)
    ],
    final_estimator=LogisticRegression()
)

stacking_ensemble.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

y_pred_stack = stacking_ensemble.predict(X_test)
print("Classification Report (Stacking):")
print(classification_report(y_test, y_pred_stack))

print("Confusion Matrix (Stacking):")
print(confusion_matrix(y_test, y_pred_stack))

y_pred_stack = stacking_ensemble.predict(X_test)
akurasi_stack = accuracy_score(y_test, y_pred_stack)

model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)
akurasi_knn = accuracy_score(y_test, y_pred_knn)

model_nb.fit(X_train, y_train)
y_pred_nb = model_nb.predict(X_test)
akurasi_nb = accuracy_score(y_test, y_pred_nb)

model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
akurasi_dt = accuracy_score(y_test, y_pred_dt)

if __name__ == "__main__":
    tampilkan_informasi_data(data)
    print(f"")

print(f"Akurasi KNN: {akurasi_knn * 100:.2f}%")
print(f"Akurasi Naive Bayes: {akurasi_nb * 100:.2f}%")
print(f"Akurasi Decision Tree: {akurasi_dt * 100:.2f}%")
print(f"Akurasi Stacking Ensemble: {akurasi_stack * 100:.2f}%")

labels = ['KNN', 'Naive Bayes', 'Decision Tree', 'Stacking Ensemble']
accuracies = [akurasi_knn * 100, akurasi_nb * 100, akurasi_dt * 100, akurasi_stack * 100]

plt.figure(figsize=(8, 5))
plt.bar(labels, accuracies, color=['blue', 'green', 'orange', 'purple'])
plt.title('Perbandingan Akurasi Model')
plt.xlabel('Model')
plt.ylabel('Akurasi (%)')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f'{acc:.2f}%', ha='center', va='bottom')

labels = ['Terkena (1)', 'Tidak Terkena (0)']
colors = ['lightblue', 'lightcoral']
plt.figure(figsize=(6, 6))
plt.pie(heart_disease_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90, shadow=True)
plt.title('Distribusi HeartDisease di Dataset')

def prediksi_manual():
    print("\nMasukkan data pasien:")
    input_data = {
        'Age': float(input("Umur (contoh: 45): ")),
        'Sex': int(input("Jenis Kelamin (0 untuk Perempuan, 1 untuk Laki-laki): ")),
        'ChestPainType': int(input("Jenis Nyeri Dada (0-3): ")),
        'RestingBP': float(input("Tekanan Darah Istirahat (contoh: 120): ")),
        'Cholesterol': float(input("Kolesterol (contoh: 200): ")),
        'FastingBS': int(input("Gula Darah Puasa > 120 mg/dl (0 atau 1): ")),
        'RestingECG': int(input("Hasil EKG Istirahat (0-2): ")),
        'MaxHR': float(input("Denyut Jantung Maksimum (contoh: 150): ")),
        'ExerciseAngina': int(input("Angina akibat Olahraga (0 untuk Tidak, 1 untuk Ya): ")),
        'Oldpeak': float(input("Depresi ST akibat olahraga (contoh: 2.0): ")),
        'ST_Slope': int(input("Kemiringan ST (0-2): "))
    }

    input_df = pd.DataFrame([input_data])
    input_df[fitur_numerik] = scaler.transform(input_df[fitur_numerik])

    hasil_probabilitas = stacking_ensemble.predict_proba(input_df)
    print(f"\nProbabilitas prediksi: {hasil_probabilitas}")

    custom_threshold = 0.6
    if hasil_probabilitas[0][1] >= custom_threshold:
        hasil = "Terkena Heart Disease"
    else:
        hasil = "Tidak Terkena Heart Disease"
    print(f"\nHasil prediksi berdasarkan ambang batas {custom_threshold}: {hasil}")


if __name__ == "__main__":
    prediksi_manual()
plt.show()