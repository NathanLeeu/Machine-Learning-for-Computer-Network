import pandas as pd

# Đọc dữ liệu từ file .txt
data = pd.read_csv(r'C:\Users\xuann\PycharmProjects\TTTN\Dataset.txt', header=None, delimiter=',')

# Tạo danh sách các kiểu tấn công
attack_types = set()

# Duyệt qua từng dòng để xác định kiểu tấn công
for index, row in data.iterrows():
    attack_type = row.iloc[-1]  # Kiểu tấn công nằm ở cột cuối cùng
    attack_types.add(attack_type)

# In ra các kiểu tấn công duy nhất
print("Các kiểu tấn công có trong dữ liệu:")
for attack in attack_types:
    print(attack)
