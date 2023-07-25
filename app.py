from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import yaml
import joblib

webapp_root = "webapp"
params_path = "params.yaml"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)


class NotANumber(Exception):
    def __init__(self, message="Values entered are not Numerical"):
        self.message = message
        super().__init__(self.message)


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def predict(data):
    config = read_params(params_path)
    model_dir_path = config["model_webapp_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data).tolist()[0]
    return prediction


def validate_input(dict_request):
    for _, val in dict_request.items():
        try:
            val = float(val)
        except Exception as e:
            raise NotANumber
    return True


def convert_to_numerical(data):
    # Convert categorical feature values to numerical values
    data["SEX"] = float(data["SEX"])  # Assuming 1=male, 2=female
    data["EDUCATION"] = float(data["EDUCATION"])  # Convert to numerical values as needed
    data["MARRIAGE"] = float(data["MARRIAGE"])  # Convert to numerical values as needed
    data["PAY_0"] = float(data["PAY_0"])  # Convert to numerical values as needed
    data["PAY_2"] = float(data["PAY_2"])
    data["PAY_3"] = float(data["PAY_3"])
    data["PAY_4"] = float(data["PAY_4"])
    data["PAY_5"] = float(data["PAY_5"])
    data["PAY_6"] = float(data["PAY_6"])
    return data


def form_response(dict_request):
    try:
        if validate_input(dict_request):
            # Convert categorical feature values to numerical values
            dict_request = convert_to_numerical(dict_request)
            data = dict_request.values()
            data = [list(map(float, data))]
            response = predict(data)

            if response == 1:
                return f"1: The customer is going to default payment next month"
            elif response == 0:
                return f"0: The customer is not going to default payment next month"
            else:
                return f"Unknown prediction result"

    except NotANumber as e:
        response = str(e)
        return response


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if request.form:
                dict_req = dict(request.form)
                response = form_response(dict_req)
                return render_template("index.html", response=response)
        except Exception as e:
            print(e)
            error = {"error": "Something went wrong!! Try again later!"}
            error = {"error": e}
            return render_template("404.html", error=error)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
