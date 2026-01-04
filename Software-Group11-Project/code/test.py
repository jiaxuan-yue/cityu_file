from flask import Flask, render_template_string

# 创建 Flask 应用
app = Flask(__name__)

# 在这里定义一个简单的 HTML 页面作为字符串，内嵌在 Python 文件中
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flask Web App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
            text-align: center;
            margin-top: 50px;
        }
        h1 {
            color: #333;
        }
        p {
            font-size: 1.2em;
            color: #555;
        }
        .btn {
            padding: 10px 20px;
            margin-top: 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 1.1em;
        }
        .btn:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>

    <h1>Welcome to My Flask Web App</h1>
    <p>This is a simple Flask app that displays an HTML page.</p>
    <button class="btn" onclick="alert('Hello, Flask!')">Click Me</button>

</body>
</html>
"""

# 定义首页路由
@app.route('/')
def index():
    return render_template_string(html_content)

# 启动 Flask 应用
if __name__ == "__main__":
    app.run(debug=True)
