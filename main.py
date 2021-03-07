from flask import Flask, request, session, render_template, redirect
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

#Overview page
@app.route("/", methods=["GET"])
def overview():
    return render_template("overview.html", results=session.get("results", None))

#About page
@app.route("/about", methods=["GET"])
def about():
    return app.send_static_file("about.html")

#Image post API endpoint
@app.route("/measure", methods=["POST"])
def measurePost():
    if "file" in request.files:
        file = request.files["file"]

        #Some file checking
        if file and (not file.filename == ""):
            if ("." in file.filename) and (file.filename.rsplit(".", 1)[1].lower() in {"png", "jpg", "jpeg", "gif"}):
                arr = numpy.array(Image.open(file).convert('RGB'), dtype=numpy.float32)
                #Send to ML for processing
                session["results"] = int(arr[0][0][0])
                return redirect("/")
                #return {"success": True, "results": }

    return redirect("/")
    #return {"success": False}


########################################################################################
# Start Flask App
########################################################################################
if __name__ == "__main__":
    app.run(ssl_context="adhoc", host="localhost", port=5000, debug=True)
