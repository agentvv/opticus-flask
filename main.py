from flask import Flask, request, abort, redirect
from PIL import Image
import numpy


########################################################################################
# App Setup
########################################################################################
app = Flask("Opticus")
app.secret_key = "Not really very secret"


########################################################################################
# Flask Routing
########################################################################################

#Home page
@app.route("/", methods=["GET"])
def index():
    return app.send_static_file("index.html")

#Image post API
@app.route("/", methods=["POST"])
def imagePost():
    if "file" in request.files:
        file = request.files["file"]

        #Some file checking
        if file and (not file.filename == ""):
            if ("." in file.filename) and (file.filename.rsplit(".", 1)[1].lower() in {"png", "jpg", "jpeg", "gif"}):
                #Send to ML for processing
                print(numpy.array(Image.open(file).convert('RGB'), dtype=numpy.float32))
                return "Success!<br/><a href=\"\\\">Go back</a>"

    return "Failure, please upload an image file<br/><a href=\"\\\">Go back</a>"


########################################################################################
# Start Flask App
########################################################################################
if __name__ == "__main__":
    app.run(ssl_context="adhoc", host="localhost", port=5000, debug=True)
