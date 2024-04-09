from flask import Flask, render_template, request, redirect,jsonify
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__, template_folder='E:\\BTX\\AI\\webtest\\templates')

# Đọc dữ liệu và huấn luyện mô hình khi ứng dụng khởi động
data_train = pd.read_csv("E:\\BTX\\AI\\webtest\\file\\Dự đoán bóng đá.csv")
data = data_train.values
attribute_data = data[:, 0:2]
label_data = data[:, 2]

convert_dataY = preprocessing.LabelEncoder()
convert_dataY.fit(label_data)
Y_train = convert_dataY.transform(label_data)

convert_dataX = preprocessing.OrdinalEncoder()
convert_dataX.fit(attribute_data)
X_train = convert_dataX.transform(attribute_data)

Dtree_Model = DecisionTreeClassifier()
Dtree_Model.fit(X_train, Y_train)


@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    home_team = request.form["home_team"]
    away_team = request.form["away_team"]

    # Kiểm tra nếu hai đội không giống nhau
    if home_team != away_team:
        # Chuyển đổi tên đội bóng thành giá trị số sử dụng ordinal encoder
        X_test = convert_dataX.transform([[home_team, away_team]])
        # Dự đoán kết quả sử dụng mô hình Decision Tree
        Y_test = Dtree_Model.predict(X_test)
        # Chuyển đổi giá trị dự đoán thành nhãn
        if Y_test[0] == 0:
            result = "Đội Nhà Thắng"
        elif Y_test[0] == 1:
            result = "Đội Nhà Thua"
        else:
            result = "Hoà"
        # Trả về kết quả dưới dạng JSON
        return f"Kết quả: {result}"
    else:
        return redirect("/")


if __name__ == "__main__":
    app.run(port=1200, debug=True)