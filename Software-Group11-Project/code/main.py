from flask import Flask, render_template, request
import os
import subprocess
import logging

# Flask 实例
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小 16MB

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 配置日志
app.logger.setLevel(logging.DEBUG)
from it_ticket import tickets_bp
app.register_blueprint(tickets_bp, url_prefix="/tickets")
@app.route("/", methods=["GET", "POST"])
def index():
    pylint_output = ""
    flake8_output = ""
    loop_output = ""
    uploaded_code = ""

    if request.method == "POST":
        app.logger.debug("Received POST request.")

        file = request.files.get("file")
        if not file:
            app.logger.error("No file uploaded.")
        else:
            app.logger.debug(f"File uploaded: {file.filename}")

        checks = request.form.getlist("check")
        app.logger.debug(f"Checks selected: {checks}")

        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            app.logger.debug(f"File saved to: {filepath}")

            # 读取上传代码内容用于高亮显示
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    uploaded_code = f.read()
            except Exception as e:
                uploaded_code = f"Error reading file: {e}"

            # 运行 Pylint
            if "pylint" in checks:
                try:
                    app.logger.debug("Running pylint...")
                    pylint_output = subprocess.run(
                        ["pylint", filepath],
                        capture_output=True, text=True, check=False
                    ).stdout
                    app.logger.debug(f"Pylint output: {pylint_output}")
                except Exception as e:
                    pylint_output = str(e)
                    app.logger.error(f"Error running pylint: {e}")

            # 运行 Flake8
            if "flake8" in checks:
                try:
                    app.logger.debug("Running flake8...")
                    flake8_output = subprocess.run(
                        ["flake8", filepath],
                        capture_output=True, text=True, check=False
                    ).stdout
                    app.logger.debug(f"Flake8 output: {flake8_output}")
                except Exception as e:
                    flake8_output = str(e)
                    app.logger.error(f"Error running flake8: {e}")

            # 简单循环嵌套检查
            if "loops" in checks:
                try:
                    app.logger.debug("Analyzing loops...")
                    loop_output = analyze_loops(filepath)
                    app.logger.debug(f"Loop analysis output: {loop_output}")
                except Exception as e:
                    loop_output = str(e)
                    app.logger.error(f"Error analyzing loops: {e}")

    return render_template(
        "index.html",
        output=True,
        pylint_output=pylint_output,
        flake8_output=flake8_output,
        loop_output=loop_output,
        uploaded_code=uploaded_code
    )

def analyze_loops(filepath):
    """简单分析 Python 文件中的循环嵌套层数"""
    max_depth = 0
    current_depth = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith(("for ", "while ")):
                current_depth += 1
                if current_depth > max_depth:
                    max_depth = current_depth
            elif stripped == "" or stripped.startswith("#"):
                continue
            elif current_depth > 0 and not stripped.startswith(("for ", "while ", "if ", "elif ", "else", "def ", "class ")):
                current_depth -= 1
    return f"最大循环嵌套层数: {max_depth}"

# 主程序入口
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8200, debug=True)
    
