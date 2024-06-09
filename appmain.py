from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/data")
def data():
    return render_template("data.html")

@app.route("/model")
def model():
    return render_template("modell.html")

@app.route("/modelexecutor")
def modelexecutor():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
