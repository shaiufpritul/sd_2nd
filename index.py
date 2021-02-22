from flask import Flask, redirect, url_for, request, render_template
from sklearn.neighbors import KNeighborsClassifier
import pickle

app = Flask(__name__)
filename = 'iris.sav'
loaded_model = pickle.load(open(filename, 'rb'))


@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/', defaults={'result': "Unavailable!"})
@app.route('/<result>')
def mainpage(result):
    return render_template("index.html", result = result)


@app.route('/submit',methods = ['GET'])
def submit():
   if request.method == 'GET':
        sl = request.args['SL']
        sw = request.args['SW']
        pl = request.args['PL']
        pw = request.args['PW']
        global loaded_model
        lb = ["Iris Setosa","Iris Versicolour","Iris Virginica"]
        ans = lb[loaded_model.predict([[sl,sw,pl,pw]])[0]]
        return redirect(url_for('mainpage',result = "It is {}!".format(ans)))
   else:
        return redirect(url_for('mainpage',result = "Something went wrong!"))

if __name__ == '__main__':
    app.run()
