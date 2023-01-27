import pickle
# from wsgiref import simple_server
from flask import Flask, request, app
from flask import Response
# from flask_cors import CORS
# import pandas as pd
from decision_tree_deploy import predObj

#  importing the necessary dependencies
from flask import Flask, render_template, request
from flask_cors import cross_origin
import pickle


application = Flask(__name__)
# CORS(app)
# app.config['DEBUG'] = True

@application.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")


@application.route("/predict_api", methods=['POST'])
def predictRoute():
    try:
        if request.json['data'] is not None:
            pred2 = predObj()
            data = request.json['data']
            print('data is:     ', data)
            res = pred2.predict_log(data)
            print('result is        ',res)
            return Response(res)
    except ValueError:
        return Response("Value not found")
    except Exception as e:
        print('exception is   ',e)
        return Response(e)


@application.route("/predict", methods=['POST','GET'])
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Pclass=float(request.form['Pclass'])
            Sex = float(request.form['Sex'])
            Age = float(request.form['Age'])
            SibSp = float(request.form['SibSp'])
            Parch = float(request.form['Parch'])
            Fare = float(request.form['Fare'])

            predict_dict ={"Pclass": Pclass,
                "Sex": Sex,
                "Age": Age,
                "SibSp": SibSp,
                "Parch":Parch,
                "Fare":Fare
    }       

            pred2 = predObj()
            res = pred2.predict_log(predict_dict)
            print('result is        ',res)
            # showing the diagnosed results in a UI
            return render_template('results.html',prediction=res)
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')


if __name__ == "__main__":
    # host = '0.0.0.0'
    # port = 5000
    application.run(debug=True)
    #httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    #httpd.serve_forever()