import json
import os
from flask import Flask, Blueprint, render_template, request, redirect, jsonify

tickets_bp = Blueprint("tickets", __name__)

# ---------------- 数据路径 ----------------
DATA_DIR = "data"
USERS_FILE = os.path.join(DATA_DIR, "users.json")
TASKS_FILE = os.path.join(DATA_DIR, "tasks.json")
REQUESTS_FILE = os.path.join(DATA_DIR, "requests.json")

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
for f in [USERS_FILE, TASKS_FILE, REQUESTS_FILE]:
    if not os.path.exists(f):
        with open(f, "w", encoding="utf-8") as fp:
            json.dump([], fp, ensure_ascii=False)

# ---------------- 工具函数 ----------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# ---------------- 首页 ----------------
@tickets_bp.route("/")
def home():
    users = load_json(USERS_FILE)
    return render_template("tickets_home.html", users=users)

# ---------------- 管理员功能 ----------------
@tickets_bp.route("/admin")
def admin_panel():
    users = load_json(USERS_FILE)
    return render_template("admin_panel.html", users=users)

@tickets_bp.route("/admin/create_user", methods=["POST"])
def create_user():
    # 支持 JSON 或表单
    if request.is_json:
        data = request.get_json()
        username = data.get("username")
    else:
        username = request.form.get("username")

    if not username:
        return jsonify({"success": False})

    users = load_json(USERS_FILE)
    if username not in users:
        users.append(username)
        save_json(USERS_FILE, users)

    return jsonify({"success": True})


@tickets_bp.route("/admin/add_task", methods=["POST"])
def add_task():
    if request.is_json:
        data = request.get_json()
        title = data.get("title")
        assigned_to = data.get("assigned_to")
        priority = data.get("priority")
    else:
        title = request.form.get("title")
        assigned_to = request.form.get("assigned_to")
        priority = request.form.get("priority")

    if not title or not assigned_to:
        return jsonify({"success": False})

    tasks = load_json(TASKS_FILE)
    tasks.append({
        "id": len(tasks)+1,
        "title": title,
        "assigned_to": assigned_to,
        "priority": int(priority)
    })
    save_json(TASKS_FILE, tasks)
    return jsonify({"success": True})

@tickets_bp.route("/admin/tasks")
def view_tasks():
    tasks = load_json(TASKS_FILE)
    # 按优先级升序排列
    tasks = sorted(tasks, key=lambda t: t["priority"])
    return render_template("tasks_list.html", tasks=tasks)

@tickets_bp.route("/admin/delete_task/<int:task_id>")
def delete_task(task_id):
    tasks = load_json(TASKS_FILE)
    tasks = [t for t in tasks if t["id"] != task_id]
    save_json(TASKS_FILE, tasks)
    return jsonify({"success": True})

# 修改优先级
@tickets_bp.route("/admin/change_priority/<int:task_id>")
def change_priority(task_id):
    tasks = load_json(TASKS_FILE)
    for t in tasks:
        if t["id"] == task_id:
            t["priority"] = t["priority"] % 3 + 1
            break
    save_json(TASKS_FILE, tasks)
    return jsonify({"success": True})

@tickets_bp.route("/admin/requests")
def view_requests():
    reqs = load_json(REQUESTS_FILE)
    pending_reqs = [r for r in reqs if r['status'] == 'pending']
    return render_template("requests_list.html", reqs=pending_reqs)

# 删除用户请求（Handled）
@tickets_bp.route("/admin/handle_request/<int:req_id>")
def handle_request(req_id):
    reqs = load_json(REQUESTS_FILE)
    reqs = [r for r in reqs if r["id"] != req_id]
    save_json(REQUESTS_FILE, reqs)
    return jsonify({"success": True})


# ---------------- 用户功能 ----------------
@tickets_bp.route("/user/<username>")
def user_panel(username):
    tasks = load_json(TASKS_FILE)
    my_tasks = [t for t in tasks if t["assigned_to"] == username]
    # 按优先级排序
    my_tasks = sorted(my_tasks, key=lambda t: t["priority"])
    reqs = load_json(REQUESTS_FILE)
    return render_template("user_panel.html", username=username, tasks=my_tasks, requests=reqs)

# 用户请求删除任务
@tickets_bp.route("/user/request_delete/<username>/<int:task_id>")
def user_request_delete(username, task_id):
    reqs = load_json(REQUESTS_FILE)
    if not any(r for r in reqs if r['username']==username and r['task_id']==task_id and r['request_type']=='delete' and r['status']=='pending'):
        new_id = max([r['id'] for r in reqs], default=0) + 1
        reqs.append({
            "id": new_id,
            "username": username,
            "task_id": task_id,
            "request_type": "delete",
            "status": "pending"
        })
        save_json(REQUESTS_FILE, reqs)
    return jsonify({"success": True})

# 用户请求修改优先级
@tickets_bp.route("/user/request_priority/<username>/<int:task_id>")
def user_request_priority(username, task_id):
    reqs = load_json(REQUESTS_FILE)
    if not any(r for r in reqs if r['username']==username and r['task_id']==task_id and r['request_type']=='priority' and r['status']=='pending'):
        new_id = max([r['id'] for r in reqs], default=0) + 1
        reqs.append({
            "id": new_id,
            "username": username,
            "task_id": task_id,
            "request_type": "priority",
            "status": "pending"
        })
        save_json(REQUESTS_FILE, reqs)
    return jsonify({"success": True})

