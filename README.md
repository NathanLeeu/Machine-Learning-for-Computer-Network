# Hướng dẫn sử dụng 
Nội dung thực tập chính:
 -   Nghiên cứu mô hình hệ thống IDPS cho mạng Hybrid sử dụng công nghệ Machine Learning
Sau đây là mô hình cơ bản của mô ML cho hệ thống IDPS:
Chuẩn bị môi trường cơ bản:
1. pandas:
Được sử dụng để xử lý dữ liệu và thao tác với DataFrame.
Cài đặt: pip install pandas
2. numpy:
Cung cấp các chức năng toán học cơ bản và làm việc với mảng số học.
Cài đặt: pip install numpy
3. scikit-learn:
Thư viện cơ bản cho các thuật toán học máy, tiền xử lý dữ liệu, và đánh giá mô hình.
Cài đặt: pip install scikit-learn
4. imblearn:
Thư viện để xử lý vấn đề dữ liệu không cân bằng, ví dụ như SMOTE.
Cài đặt: pip install imbalanced-learn
5. matplotlib:
Dùng để vẽ đồ thị và biểu đồ.
Cài đặt: pip install matplotlib
6. seaborn:
Thư viện vẽ đồ thị dựa trên matplotlib, cung cấp các chức năng để tạo biểu đồ đẹp và dễ hiểu hơn.
Cài đặt: pip install seaborn
Tổng hợp lệnh cài như sau:

pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn notebook

Sau đó đi từng bước để hoàn thiện mô hình:

Bước 1: Thu Thập và Chuẩn Bị Dữ Liệu
Trong bước này, chúng ta sẽ thực hiện các nhiệm vụ sau:
- Tải Dữ Liệu: Sử dụng pandas để tải dữ liệu từ một URL.
-  Mã Hóa Dữ Liệu: Chuyển đổi các cột không phải số sang dạng số.
- Xử Lý Nhãn: Mã hóa nhãn thành các loại tấn công cụ thể.
- Chuẩn Hóa Dữ Liệu: Sử dụng StandardScaler để chuẩn hóa dữ liệu.
- Cân Bằng Dữ Liệu: Sử dụng SMOTE để cân bằng các lớp dữ liệu.
- Chia Dữ Liệu: Chia dữ liệu thành tập huấn luyện và kiểm tra.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import logging

# Cấu hình ghi log
logging.basicConfig(filename='model_training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
error_log_file = 'error_log.txt'

def log_error(message):
    with open(error_log_file, 'a') as f:
        f.write(f"{message}\n")
    logging.error(message)

def load_and_preprocess_data():
    try:
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
        for col in categorical_cols:
            data[col] = data[col].astype('category').cat.codes

        # Phân chia dữ liệu thành features và labels
        x = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # Mã hóa nhãn thành các loại tấn công cụ thể
        attack_types = {
            0: 'normal',
            1: 'DoS', 2: 'DoS', 3: 'DoS', 4: 'DoS', 5: 'DoS', 6: 'DoS',
            7: 'U2R', 8: 'U2R', 9: 'U2R', 10: 'U2R',
            11: 'R2L', 12: 'R2L', 13: 'R2L', 14: 'R2L', 15: 'R2L', 16: 'R2L', 17: 'R2L', 18: 'R2L',
            19: 'Probe', 20: 'Probe', 21: 'Probe', 22: 'Probe'
        }

        # Thay đổi cách mã hóa nhãn
        y_mapped = [attack_types.get(label, 'unknown') for label in y]

        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        # Cân bằng dữ liệu
        smote = SMOTE(random_state=42)
        x, y_mapped = smote.fit_resample(x, y_mapped)

        # Chia dữ liệu thành tập huấn luyện và kiểm tra
        x_train, x_test, y_train, y_test = train_test_split(x, y_mapped, test_size=0.3, random_state=42,
                                                            stratify=y_mapped)
        logging.info("Data preprocessing complete.")
        return x_train, x_test, y_train, y_test, scaler

    except Exception as e:
        log_error(f"Error in load_and_preprocess_data: {str(e)}")
        raise

# Chạy hàm chuẩn bị dữ liệu
x_train, x_test, y_train, y_test, scaler = load_and_preprocess_data()

Phân Tích Bước 1:
- Tải Dữ Liệu: Chúng ta sử dụng pandas để tải dữ liệu từ một liên kết công khai. Dữ liệu này chứa các đặc trưng của các phiên mạng và nhãn chỉ loại tấn công.
- Mã Hóa Dữ Liệu: Các cột phân loại như protocol_type, service, và flag được mã hóa thành số để dễ dàng xử lý bởi các mô hình machine learning.
- Xử Lý Nhãn: Nhãn được chuyển thành các loại tấn công cụ thể để mô hình dễ phân loại.
- Chuẩn Hóa Dữ Liệu: StandardScaler được sử dụng để chuẩn hóa dữ liệu, đảm bảo rằng tất cả các đặc trưng có cùng quy mô.
- Cân Bằng Dữ Liệu: SMOTE giúp cân bằng số lượng mẫu giữa các lớp để cải thiện hiệu suất của mô hình.
- Chia Dữ Liệu: Dữ liệu được chia thành tập huấn luyện và kiểm tra, với tỉ lệ 70% cho huấn luyện và 30% cho kiểm tra.

Bước 2: Xây Dựng và Huấn Luyện Các Mô Hình Machine Learning
Trong bước này, chúng ta sẽ:
- Chọn Các Mô Hình: Lựa chọn và cấu hình các mô hình machine learning phổ biến như K-Nearest Neighbors, Support Vector Machine, Decision Tree, Random Forest, Neural Networks, Logistic Regression, và Gradient Boosting.
- Huấn Luyện Các Mô Hình: Sử dụng GridSearchCV để tìm kiếm các siêu tham số tối ưu cho từng mô hình.
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

def train_models(x_train, y_train):
    try:
        models = {}

        # K-Nearest Neighbors
        knn_params = {
            'n_neighbors': [3, 5, 7, 9],
            'metric': ['euclidean', 'manhattan']
        }
        knn = KNeighborsClassifier()
        knn_grid = GridSearchCV(knn, knn_params, cv=5)
        knn_grid.fit(x_train, y_train)
        models['KNN'] = knn_grid.best_estimator_
        logging.info("KNN model trained.")

        # Support Vector Machine
        svm_params = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1],
            'kernel': ['linear', 'rbf']
        }
        svm = SVC(probability=True)
        svm_grid = GridSearchCV(svm, svm_params, cv=5)
        svm_grid.fit(x_train, y_train)
        models['SVM'] = svm_grid.best_estimator_
        logging.info("SVM model trained.")

        # Decision Tree
        dt_params = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        dt = DecisionTreeClassifier()
        dt_grid = GridSearchCV(dt, dt_params, cv=5)
        dt_grid.fit(x_train, y_train)
        models['Decision Tree'] = dt_grid.best_estimator_
        logging.info("Decision Tree model trained.")

        # Random Forest
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        rf = RandomForestClassifier()
        rf_grid = GridSearchCV(rf, rf_params, cv=5)
        rf_grid.fit(x_train, y_train)
        models['Random Forest'] = rf_grid.best_estimator_
        logging.info("Random Forest model trained.")

        # Neural Networks
        mlp_params = {
            'hidden_layer_sizes': [(50,), (100,), (100, 100)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd']
        }
        mlp = MLPClassifier(max_iter=300)
        mlp_grid = GridSearchCV(mlp, mlp_params, cv=5)
        mlp_grid.fit(x_train, y_train)
        models['Neural Networks'] = mlp_grid.best_estimator_
        logging.info("Neural Networks model trained.")

        # Logistic Regression
        lr_params = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2']
        }
        lr = LogisticRegression(max_iter=300)
        lr_grid = GridSearchCV(lr, lr_params, cv=5)
        lr_grid.fit(x_train, y_train)
        models['Logistic Regression'] = lr_grid.best_estimator_
        logging.info("Logistic Regression model trained.")

        # Gradient Boosting
        gb_params = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1]
        }
        gb = GradientBoostingClassifier()
        gb_grid = GridSearchCV(gb, gb_params, cv=5)
        gb_grid.fit(x_train, y_train)
        models['Gradient Boosting'] = gb_grid.best_estimator_
        logging.info("Gradient Boosting model trained.")

        return models

    except Exception as e:
        log_error(f"Error in train_models: {str(e)}")
        raise

# Huấn luyện các mô hình
models = train_models(x_train, y_train)

Phân Tích Bước 2
- Chọn Các Mô Hình: Chúng ta đã chọn một loạt các mô hình machine learning phổ biến và cấu hình các siêu tham số mà chúng ta muốn tối ưu hóa.
- Huấn Luyện Các Mô Hình: GridSearchCV giúp chúng ta tìm ra các giá trị siêu tham số tốt nhất cho mỗi mô hình bằng cách thực hiện tìm kiếm lưới (grid search) và đánh giá hiệu suất mô hình trên tập dữ liệu huấn luyện
Bước 3: Đánh Giá Các Mô Hình
Trong bước này, chúng ta sẽ:
- Dự Đoán và Đánh Giá: Sử dụng các mô hình để dự đoán trên dữ liệu kiểm tra và tính toán các chỉ số đánh giá như accuracy, classification report, confusion matrix, và ROC AUC.
- Vẽ Biểu Đồ: Vẽ confusion matrix và ROC curve để trực quan hóa hiệu suất của mô hình.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

def evaluate_models(models, x_test, y_test):
    try:
        for model_name, model in models.items():
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(x_test)[:, 1], pos_label='normal')
            roc_auc = auc(fpr, tpr)

            logging.info(f"{model_name} Accuracy: {accuracy}")
            logging.info(f"{model_name} Classification Report:\n{report}")
            logging.info(f"{model_name} Confusion Matrix:\n{cm}")
            logging.info(f"{model_name} ROC AUC: {roc_auc}")

            # Vẽ Confusion Matrix
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.savefig(f'{model_name}_confusion_matrix.png')
            plt.close()

            # Vẽ ROC Curve
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Receiver Operating Characteristic - {model_name}')
            plt.legend(loc='lower right')
            plt.savefig(f'{model_name}_roc_curve.png')
            plt.close()

    except Exception as e:
        log_error(f"Error in evaluate_models: {str(e)}")
        raise

# Đánh giá các mô hình
evaluate_models(models, x_test, y_test)

Phân Tích Bước 3
- Dự Đoán và Đánh Giá: Chúng ta tính toán các chỉ số đánh giá mô hình và ghi lại chúng vào file log. Các chỉ số này giúp chúng ta hiểu được hiệu suất của mô hình trên tập dữ liệu kiểm tra.
- Vẽ Biểu Đồ: Các biểu đồ như confusion matrix và ROC curve giúp trực quan hóa hiệu suất của các mô hình, cho phép chúng ta so sánh giữa các mô hình dễ dàng hơn.
Bước 4: Phát Hiện Xâm Nhập
Trong bước này, chúng ta sẽ:
- Phát Hiện Xâm Nhập: Sử dụng mô hình đã huấn luyện để dự đoán loại tấn công trên dữ liệu mới.
def detect_intrusion(model, scaler, new_data):
    try:
        new_data = scaler.transform(new_data)
        predictions = model.predict(new_data)
        return predictions
    except Exception as e:
        log_error(f"Error in detect_intrusion: {str(e)}")
        raise

# Ví dụ phát hiện xâm nhập
sample_data = np.array([[0.1, 0.2, 0.3, ...]])  # Thay thế bằng dữ liệu mới
predictions = detect_intrusion(models['KNN'], scaler, sample_data)
print("Intrusion Detection Predictions:", predictions)

