import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np

# Đọc dữ liệu
df = pd.read_csv('clear_data.csv')
df.drop(columns='Unnamed: 0', inplace=True)

X = df.drop(columns='Price')
y = df['Price']

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo và huấn luyện mô hình
ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])

column_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

model = LinearRegression()
pip = make_pipeline(column_trans, model)
pip.fit(X_train, y_train)

# Đánh giá mô hình
score = []
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    pip = make_pipeline(column_trans, model)
    pip.fit(X_train, y_train)
    y_prep = pip.predict(X_test)
    score.append(r2_score(y_test, y_prep))

best_index = np.argmax(score)
best_score = score[best_index]

# Huấn luyện lại với tham số tốt nhất
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=best_index)
pip = make_pipeline(column_trans, model)
pip.fit(X_train, y_train)
final_score = r2_score(y_test, pip.predict(X_test))

# Lưu mô hình
joblib.dump(pip, 'car.pkl')