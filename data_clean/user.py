from flask import Flask, request, jsonify
import secrets
import string
import csv
import os

app = Flask(__name__)

# 定义生成随机字符串的函数
def generate_random_string():
    allowed_chars = string.ascii_letters + string.digits
    return ''.join(secrets.choice(allowed_chars) for _ in range(16))

def save_to_csv(data, out_dir, filename='user_ids.csv'):
    output_file_path = os.path.join(out_dir, filename)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(output_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['User ID'])

        for user_id in data:
            writer.writerow([user_id])

@app.route('/generate_user_ids', methods=['POST'])
def generate_user_ids_api():
    try:
        num_users = int(request.json['num_users'])
        out_dir = request.json['out_dir']
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Invalid or missing 'num_users' parameter. Please provide an integer value."}), 400

    if num_users <= 0 or num_users > 1000:  # 假设最大允许生成1000个用户ID
        return jsonify({"error": f"Number of user IDs requested ({num_users}) is outside the allowed range (1-1000)."}), 400

    users = [generate_random_string() for _ in range(num_users)]

    # 保存用户ID到CSV文件
    save_to_csv(users, out_dir)

    return jsonify({"user_ids": users}), 200

if __name__ == '__main__':
    app.run(debug=True)