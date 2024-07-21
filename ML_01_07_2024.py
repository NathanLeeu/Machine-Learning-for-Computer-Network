import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


# Bước 1: Thu Thập và Chuẩn Bị Dữ Liệu
def load_and_preprocess_data():
    url = 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt'
    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'label'
    ]
    data = pd.read_csv(url, names=column_names)

    # Mã hóa các cột không phải số
    categorical_cols = ['protocol_type', 'service', 'flag']
    encoder = LabelEncoder()
    for col in categorical_cols:
        data[col] = encoder.fit_transform(data[col])

    # Phân chia dữ liệu thành features và labels
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Mã hóa nhãn thành các loại tấn công cụ thể
    attack_types = {
        'normal': 'normal',
        'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS',
        'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
        'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L',
        'warezclient': 'R2L', 'warezmaster': 'R2L',
        'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe'
    }
    y = np.array([attack_types[label] for label in y])

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test, scaler


# Bước 2: Xây Dựng và Huấn Luyện Các Mô Hình Machine Learning
def train_models(x_train, y_train):
    models = {}

    # K-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    models['KNN'] = knn

    # Support Vector Machine
    svm = SVC(probability=True)
    svm.fit(x_train, y_train)
    models['SVM'] = svm

    # Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    models['Decision Tree'] = dt

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(x_train, y_train)
    models['Random Forest'] = rf

    # Neural Networks
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
    mlp.fit(x_train, y_train)
    models['Neural Networks'] = mlp

    # Logistic Regression
    lr = LogisticRegression(max_iter=300)
    lr.fit(x_train, y_train)
    models['Logistic Regression'] = lr

    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100)
    gb.fit(x_train, y_train)
    models['Gradient Boosting'] = gb

    return models


# Bước 3: Đánh Giá Các Mô Hình
def evaluate_models(models, x_test, y_test):
    for model_name, model in models.items():
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {accuracy:.2f}")
        print(classification_report(y_test, y_pred))
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d')
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        # Vẽ ROC Curve và tính AUC
        if model_name in ['KNN', 'SVM', 'Random Forest', 'Neural Networks', 'Logistic Regression', 'Gradient Boosting']:
            y_proba = model.predict_proba(x_test)
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label='DoS')
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} ROC Curve')
            plt.legend(loc="lower right")
            plt.show()


# Bước 4: Triển Khai Hệ Thống Đơn Giản (ví dụ với KNN)
def detect_intrusion(model, scaler, data_point):
    data_point = scaler.transform([data_point])
    prediction = model.predict(data_point)
    return prediction[0]


def main():
    x_train, x_test, y_train, y_test, scaler = load_and_preprocess_data()
    models = train_models(x_train, y_train)
    evaluate_models(models, x_test, y_test)

    # Ví dụ phát hiện xâm nhập sử dụng KNN
    sample_data = x_test[0]  # Lấy một điểm dữ liệu từ tập kiểm tra
    knn_model = models['KNN']
    result = detect_intrusion(knn_model, scaler, sample_data)
    print(f"Detection Result for sample data: {result}")


if __name__ == "__main__":
    main()
