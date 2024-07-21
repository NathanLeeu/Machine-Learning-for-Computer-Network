import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
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


# Bước 1: Thu Thập và Chuẩn Bị Dữ Liệu
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
        encoder = LabelEncoder()
        for col in categorical_cols:
            data[col] = encoder.fit_transform(data[col])

        # Phân chia dữ liệu thành features và labels
        x = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # In ra các giá trị duy nhất trong y trước khi mã hóa
        unique_labels = np.unique(y)
        logging.info(f"Unique values in y before mapping: {unique_labels}")

        # Mã hóa nhãn thành các loại tấn công cụ thể
        attack_types = {
            0: 'normal',
            1: 'DoS', 2: 'DoS', 3: 'DoS', 4: 'DoS', 5: 'DoS', 6: 'DoS',
            7: 'U2R', 8: 'U2R', 9: 'U2R', 10: 'U2R',
            11: 'R2L', 12: 'R2L', 13: 'R2L', 14: 'R2L', 15: 'R2L', 16: 'R2L', 17: 'R2L', 18: 'R2L',
            19: 'Probe', 20: 'Probe', 21: 'Probe', 22: 'Probe'
        }

        y_mapped = np.array([attack_types.get(label, 'unknown') for label in y])

        # Kiểm tra giá trị trong y sau khi mã hóa
        logging.info(f"Unique values in y after mapping: {np.unique(y_mapped)}")

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


# Bước 2: Xây Dựng và Huấn Luyện Các Mô Hình Machine Learning
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


# Bước 3: Đánh Giá Các Mô Hình
def evaluate_models(models, x_test, y_test):
    try:
        for model_name, model in models.items():
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"{model_name} Accuracy: {accuracy:.2f}")
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
            if model_name in ['KNN', 'SVM', 'Random Forest', 'Neural Networks', 'Logistic Regression',
                              'Gradient Boosting']:
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

        logging.info("Model evaluation complete.")

    except Exception as e:
        log_error(f"Error in evaluate_models: {str(e)}")
        raise


# Bước 4: Triển Khai Hệ Thống Đơn Giản (ví dụ với KNN)
def detect_intrusion(model, scaler, data_point):
    try:
        data_point = scaler.transform([data_point])
        prediction = model.predict(data_point)
        return prediction[0]
    except Exception as e:
        log_error(f"Error in detect_intrusion: {str(e)}")
        raise


def main():
    try:
        x_train, x_test, y_train, y_test, scaler = load_and_preprocess_data()
        models = train_models(x_train, y_train)
        evaluate_models(models, x_test, y_test)

        # Ví dụ phát hiện xâm nhập sử dụng KNN
        sample_data = x_test[0]  # Lấy một điểm dữ liệu từ tập kiểm tra
        knn_model = models['KNN']
        result = detect_intrusion(knn_model, scaler, sample_data)
        print(f"Detection Result for sample data: {result}")

        logging.info("Process complete.")

    except Exception as e:
        log_error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
