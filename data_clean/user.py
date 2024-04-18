from flask import Flask, request, jsonify
import secrets
import string

app = Flask(__name__)

# 定义生成随机字符串的函数
def generate_random_string():
    allowed_chars = string.ascii_letters + string.digits
    return ''.join(secrets.choice(allowed_chars) for _ in range(16))

@app.route('/generate_user_ids', methods=['POST'])
def generate_user_ids_api():
    # 获取请求中的用户ID数量参数
    try:
        num_users = int(request.json['num_users'])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Invalid or missing 'num_users' parameter. Please provide an integer value."}), 400

    # 检查请求的数量是否合理（根据实际需求设置合适的范围限制）
    if num_users <= 0 or num_users > 1000:  # 假设最大允许生成1000个用户ID
        return jsonify({"error": f"Number of user IDs requested ({num_users}) is outside the allowed range (1-1000)." }), 400

    # 创建用户ID列表
    users = [generate_random_string() for _ in range(num_users)]

    return jsonify({"user_ids": users}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)