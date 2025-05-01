from flask import request, render_template, url_for, redirect
import os
from app import create_app
from werkzeug.utils import secure_filename
from backend.test import test
from backend.config import cfg
from backend import results

app = create_app()


@app.route('/')
def start():
    print("start running")
    return render_template("index.html")


@app.route('/index.html')
def index():
    return render_template("index.html")


@app.route('/case.html')
def case():
    return render_template("case.html")


@app.route('/diagnosis.html')
def diagnosis():
    return render_template("diagnosis.html")


@app.route('/login.html')
def login():
    return render_template("login.html")


@app.route('/diagnosis_result.html', methods=['GET', 'POST'])
def diagnosis_result():
    # 设置一个用于上传的文件夹路径
    UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    # 确保上传文件夹存在
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    if request.method == 'POST':
        # 获取表单信息
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        dob = request.form.get('dob')
        phone = request.form.get('phone')
        file = request.files['imageUpload']

        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(save_path)
            file.save(save_path)

            # 获取预测结果（具体病种）
            prediction,confidence = test(cfg, save_path,path='backend')

            diagnosis_text = f"根据您的图片分析，您有{confidence:.2f}%的几率患有{prediction}皮肤病，建议到皮肤科进一步诊治。"
            return render_template(
                'diagnosis_result.html',
                image_url=url_for('static', filename='uploads/' + filename),
                diagnosis_text=diagnosis_text,
                name=name,
                age=age,
                gender=gender,
                dob=dob,
                phone=phone
            )
        else:
            return "无效的文件类型", 400

    return redirect(url_for('diagnosis'))

@app.route('/news.html')
def news():
    return render_template("news.html")


@app.route('/newsDetail.html')
def newsDetail():
    return render_template("newsDetail.html")


@app.route('/support.html')
def support():
    return render_template("support.html")


if __name__ == '__main__':
    app.run()
