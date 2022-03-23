from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)


@app.route("/")
@app.route("/home")
def home():
    print(url_for("home"))
    return render_template("index.html")


@app.route("/query")
def query():
    query = request.args
    return query


@app.route("/post", methods=["GET", "POST"])
def post():
    if request.method == "GET":
        return render_template("post.html")
    else:
        return request.form


@app.route("/<path>")
def wrong_urls(path):
    print(f"Redirect to '/' from '/{path}'")
    return redirect(url_for("home"))


app.run(host="0.0.0.0", port=5000, debug=True)
