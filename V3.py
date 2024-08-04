import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import logging
import time
from imblearn.over_sampling import SMOTE

# Cấu hình ghi log
logging.basicConfig(
    filename='model_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

error_log_file = 'error_log.txt'


def log_error(message):
    with open(error_log_file, 'a', encoding='utf-8') as f:
        f.write(f"{message}\n")
    logging.error(message)


def log_info(message):
    logging.info(message)
    print(message)


def identify_attack(label):
    attack_types = {
        'normal': 'Normal',
        'neptune': 'Denial of Service',
        'warezclient': 'Warez Client',
        'ipsweep': 'Port Scan',
        # Thêm các dạng tấn công khác nếu cần
    }
    return attack_types.get(label, 'Unknown')


def create_empty_files():
    """ Tạo các tệp rỗng nếu chúng không tồn tại. """
    paths = [
        'C:/Users/xuann/PycharmProjects/TTTN/Dataset.txt',
        'C:/Users/xuann/PycharmProjects/TTTN/Dataframe.txt',
        'C:/Users/xuann/PycharmProjects/TTTN/Attack_Type_Learned.txt'
    ]

    for path in paths:
        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write('')  # Tạo tệp rỗng
            log_info(f"Đã tạo tệp rỗng: {path}")


def load_and_preprocess_data():
    create_empty_files()  # Tạo các tệp cần thiết nếu chưa tồn tại

    try:
        # Đọc dữ liệu từ dataset.txt
        local_data_path = 'C:/Users/xuann/PycharmProjects/TTTN/Dataset.txt'
        if not os.path.exists(local_data_path) or os.path.getsize(local_data_path) == 0:
            log_error(f"Tệp dữ liệu '{local_data_path}' không tồn tại hoặc rỗng.")
            raise FileNotFoundError(f"Tệp dữ liệu '{local_data_path}' không tồn tại hoặc rỗng.")

        local_data = pd.read_csv(local_data_path, sep='\t')
        log_info("Đã tải dữ liệu từ dataset.txt.")
        print(local_data.head())  # Kiểm tra dữ liệu

        # Danh sách các URL hoặc đường dẫn tới các tập dữ liệu từ các kho lưu trữ khác nhau
        urls = [
            'https://raw.githubusercontent.com/username/repository1/branch/path/to/KDD_Cup_1999.csv',
            'https://raw.githubusercontent.com/username/repository2/branch/path/to/CICIDS_2017.csv',
            'https://raw.githubusercontent.com/username/repository3/branch/path/to/another_dataset.csv'
            # Thêm nhiều URL khác nếu cần
        ]

        # Thử tải dữ liệu từ URL và kết hợp vào dữ liệu địa phương
        try:
            url_data_frames = [pd.read_csv(url) for url in urls]
            url_data = pd.concat(url_data_frames, ignore_index=True)
            log_info("Đã tải dữ liệu từ các URL.")
            print(url_data.head())  # Kiểm tra dữ liệu
        except Exception as e:
            log_error(f"Lỗi khi tải dữ liệu từ URL: {str(e)}")
            url_data = pd.DataFrame()  # Tạo DataFrame rỗng để không ảnh hưởng đến việc kết hợp

        # Kết hợp dữ liệu từ cả hai nguồn
        data = pd.concat([local_data, url_data], ignore_index=True)
        log_info("Kết hợp dữ liệu thành công.")
        print(data.head())  # Kiểm tra dữ liệu

        # Đảm bảo dữ liệu có cột 'label'
        if 'label' not in data.columns:
            raise ValueError("The dataset must contain a 'label' column.")

        # Mã hóa nhãn thành các loại tấn công cụ thể
        data['attack_type'] = data['label'].apply(identify_attack)

        # Lưu DataFrame vào file 'Dataframe.txt'
        data.to_csv('C:/Users/xuann/PycharmProjects/TTTN/Dataframe.txt', index=False)

        # Tạo danh sách các kiểu tấn công đã học được
        attack_types = data['attack_type'].unique()

        # Tạo DataFrame chứa danh sách các kiểu tấn công
        attack_types_df = pd.DataFrame(attack_types, columns=['attack_type'])

        # Lưu danh sách các kiểu tấn công vào file 'Attack_Type_Learned.txt'
        attack_types_df.to_csv('C:/Users/xuann/PycharmProjects/TTTN/Attack_Type_Learned.txt', index=False)

        log_info("Đã lưu DataFrame và danh sách các kiểu tấn công.")

        # Liệt kê các cột không phải số
        categorical_cols = ['protocol_type', 'service', 'flag']  # Thay đổi theo dữ liệu cụ thể
        for col in categorical_cols:
            if col in data.columns:
                data[col] = data[col].astype('category').cat.codes

        # Chuẩn hóa dữ liệu
        features = data.drop(columns=['label', 'attack_type']).values
        labels = data['attack_type'].values
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Cân bằng dữ liệu
        smote = SMOTE(random_state=42)
        features_balanced, labels_balanced = smote.fit_resample(features_scaled, labels)

        # Chia dữ liệu thành tập huấn luyện và kiểm tra
        x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(
            features_balanced, labels_balanced, test_size=0.3, random_state=42, stratify=labels_balanced
        )
        log_info("Xử lý dữ liệu hoàn tất.")
        return x_train_data, x_test_data, y_train_data, y_test_data, scaler

    except Exception as e:
        log_error(f"Lỗi trong load_and_preprocess_data: {str(e)}")
        raise


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
        log_info("Mô hình KNN đã được huấn luyện.")

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
        log_info("Mô hình SVM đã được huấn luyện.")

        # Decision Tree
        dt_params = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        dt = DecisionTreeClassifier()
        dt_grid = GridSearchCV(dt, dt_params, cv=5)
        dt_grid.fit(x_train, y_train)
        models['Decision Tree'] = dt_grid.best_estimator_
        log_info("Mô hình Decision Tree đã được huấn luyện.")

        # Random Forest
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        rf = RandomForestClassifier()
        rf_grid = GridSearchCV(rf, rf_params, cv=5)
        rf_grid.fit(x_train, y_train)
        models['Random Forest'] = rf_grid.best_estimator_
        log_info("Mô hình Random Forest đã được huấn luyện.")

        # Gradient Boosting
        gb_params = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5]
        }
        gb = GradientBoostingClassifier()
        gb_grid = GridSearchCV(gb, gb_params, cv=5)
        gb_grid.fit(x_train, y_train)
        models['Gradient Boosting'] = gb_grid.best_estimator_
        log_info("Mô hình Gradient Boosting đã được huấn luyện.")

        # Neural Network
        nn_params = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['tanh', 'relu'],
            'solver': ['adam', 'sgd']
        }
        nn = MLPClassifier(max_iter=1000)
        nn_grid = GridSearchCV(nn, nn_params, cv=5)
        nn_grid.fit(x_train, y_train)
        models['Neural Network'] = nn_grid.best_estimator_
        log_info("Mô hình Neural Network đã được huấn luyện.")

        # Logistic Regression
        lr_params = {
            'C': [0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga']
        }
        lr = LogisticRegression()
        lr_grid = GridSearchCV(lr, lr_params, cv=5)
        lr_grid.fit(x_train, y_train)
        models['Logistic Regression'] = lr_grid.best_estimator_
        log_info("Mô hình Logistic Regression đã được huấn luyện.")

        return models

    except Exception as e:
        log_error(f"Lỗi trong train_models: {str(e)}")
        raise


def evaluate_models(models, x_test, y_test):
    try:
        for name, model in models.items():
            log_info(f"\nĐánh giá mô hình: {name}")

            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            log_info(f"{name} Accuracy: {accuracy:.4f}")

            class_report = classification_report(y_test, y_pred)
            log_info(f"{name} Classification Report:\n{class_report}")

            cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
            log_info(f"{name} Confusion Matrix:\n{cm}")

            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(x_test)[:, 1],
                                    pos_label='DoS')  # Thay đổi 'DoS' thành lớp cần đánh giá
            roc_auc = auc(fpr, tpr)
            log_info(f"\n{name} ROC AUC: {roc_auc:.4f}")

            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend(loc='lower right')
            plt.savefig(f'{name}_roc_curve.png')
            plt.close()

    except Exception as e:
        log_error(f"Lỗi trong evaluate_models: {str(e)}")
        raise


def main():
    try:
        while True:  # Vòng lặp vô hạn để chạy chương trình liên tục
            log_info("Bắt đầu quá trình huấn luyện và đánh giá mô hình.")

            x_train_data, x_test_data, y_train_data, y_test_data, scaler = load_and_preprocess_data()
            models = train_models(x_train_data, y_train_data)
            evaluate_models(models, x_test_data, y_test_data)

            log_info("Hoàn thành huấn luyện và đánh giá mô hình.")

            # Thay đổi thời gian chờ giữa các vòng lặp theo nhu cầu
            log_info("Chương trình sẽ tiếp tục chạy, nhấn Ctrl+C để dừng.")
            time.sleep(3600)  # Chờ 1 giờ trước khi lặp lại, điều chỉnh theo nhu cầu

    except KeyboardInterrupt:
        log_info("Chương trình đã bị dừng bởi người dùng.")
    except Exception as e:
        log_error(f"Lỗi trong main: {str(e)}")


if __name__ == "__main__":
    main()
