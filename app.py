import json
from datetime import timedelta, datetime
from bson import ObjectId, json_util
from flask import Flask, jsonify
import os
from dotenv import load_dotenv
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, unset_jwt_cookies, jwt_required
from pymongo import MongoClient
from flask import request
from flask_bcrypt import generate_password_hash, check_password_hash
import pickle
import pandas as pd
import numpy as np
import cv2

load_dotenv()
SECRET_KEY = os.getenv('SECRET_KEY')
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=1)
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['JWT_SECRET_KEY'] = JWT_SECRET_KEY
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = JWT_ACCESS_TOKEN_EXPIRES

jwt = JWTManager(app)

client = MongoClient(f"mongodb+srv://{DB_USERNAME}:{DB_PASSWORD}@pts.mlprddc.mongodb.net/?retryWrites=true&w=majority")
db = client.weinguard
loaded_model = pickle.load(open('random_forest.pkl', 'rb'))


@app.route('/register', methods=['POST'])
def register():
    try:
        if "username" not in request.form or "password" not in request.form:
            error = "Uporabnisko ime ali geslo je prazno."
            return {
                "error": error
            }, 400
        users_db = db.user
        username = request.form['username']
        password = request.form['password']
        hashed_pass = generate_password_hash(password)
        if not username:
            error = 'Uporabnisko ime je prazno.'
            return {
                "error": error
            }, 400
        elif not password:
            error = 'Geslo je prazno.'
            return {
                "error": error
            }, 400
        if users_db.find_one({'username': username}):
            error = "Uporabnisko ime ze obstaja."
            return {
                "error": error
            }, 400
        else:
            user_id = users_db.insert_one(
                {'username': username, 'password': hashed_pass}).inserted_id
            data = {
                "user_id": str(user_id)
            }
    except Exception as e:
        return {
            "error": e
        }, 400
    return data, 201


@app.route('/login', methods=['POST'])
def login():
    try:
        users_db = db.user
        if request.method == 'POST':
            if "username" not in request.form or "password" not in request.form:
                error = "Uporabnisko ime ali geslo je prazno."
                return {
                    "error": error
                }, 400
            username = request.form['username']
            password = request.form['password']
            user = users_db.find_one({"username": username})

            # if user is not None:
            #     hashedPass = hashing.hash_value(password, user["salt"])

            check_pwd = check_password_hash(user["password"], password)

            if user is None:
                error = 'Nepravilno uporabniško ime.'
                return {
                    "error": error
                }, 400

            if not check_pwd:
                error = 'Nepravilno geslo.'
                return {
                    "error": error
                }, 400

            payload = {
                "sub": str(user["_id"]),
                "name": username,
                "iat": str(datetime.now()),
                "exp": str(datetime.now() + timedelta(hours=2))
            }
            access_token = create_access_token(identity=payload)
            username = user['username']
            data = {
                "username": username,
                "_id": str(user["_id"]),
                "token": access_token
            }
    except Exception as e:
        return {
            "error": e
        }, 400
    return data, 200


@app.route('/logout', methods=["POST"])
def logout():
    try:
        response = jsonify({"message": "succesfully logged out"})
        unset_jwt_cookies(response)
    except Exception as e:
        return {
            "error": e
        }, 400
    return response, 200


@app.route('/users')
# @jwt_required()
def users():
    try:
        users_db = db.user
        documents = [doc for doc in
                     users_db.find({}, {"username": 1})]
    except Exception as e:
        return {
            "error": e
        }, 400
    return json_util.dumps(documents), 200


@app.route('/user/<uid>')
# @jwt_required()
def user(uid):
    try:
        if len(uid) != 24:
            error = "Napačen id."
            return {"error": error}, 400
        users_db = db.user
        user = users_db.find_one({"_id": ObjectId(uid)}, {"username": 1})
        if user is None:
            error = "Napačen id."
            return {"error": error}, 400
    except Exception as e:
        return {
            "error": e
        }, 400
    return json.loads(json_util.dumps(user)), 200


@app.route('/update/<uid>', methods=["PUT"])
def update_user(uid):
    try:
        if len(uid) != 24:
            error = "Napačen id."
            return {
                "error": error
            }, 400
        if "username" not in request.form:
            error = "Uporabnisko ime je prazno."
            return {
                "error": error
            }, 400
        users_db = db.user
        user = users_db.find_one({"_id": ObjectId(uid)})
        username = request.form['username']

        if user is None:
            error = "Uporabnik ne obstaja"
            return {
                "error": error
            }, 400
        newvalues = {"$set": {'username': username}}
        users_db.update_one({"_id": ObjectId(uid)}, newvalues)
        data = {
            "success": "true"
        }
    except Exception as e:
        return {
            "error": e
        }, 400
    return data, 200


@app.route('/delete/<uid>', methods=["DELETE"])
def delete_user(uid):
    try:
        if len(uid) != 24:
            error = "Napačen id."
            return {
                "error": error
            }, 400
        users_db = db.user
        user = users_db.find_one({"_id": ObjectId(uid)})

        if user is None:
            error = "Uporabnik ne obstaja"
            return {
                "error": error
            }, 400
        users_db.delete_one({"_id": ObjectId(uid)})
        data = {
            "success": "true"
        }
    except Exception as e:
        return {
            "error": e
        }, 400
    return data, 200


@app.route('/predict', methods=["POST"])
def predict():
    try:
        if request.method == 'POST':
            # print(request.files.get('prediction_image', ''))
            uploaded_image = request.files.get('prediction_image', '')
            image_data = uploaded_image.read()
            np_img = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            predict_img = cv2.resize(img, (128, 128)).flatten()
            feature_names = [str(i) for i in range(len(predict_img))]

            df = pd.DataFrame([predict_img], columns=feature_names)

            prediciton = loaded_model.predict(df)
            data = {"prediction": str(prediciton)}
            return jsonify(data), 200
    except Exception as e:
        return {
            "error": e
        }, 400


if __name__ == '__main__':
    app.run(debug=True)
