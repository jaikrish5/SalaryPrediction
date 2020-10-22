from flask import Flask,render_template,request,jsonify
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/', methods=['GET', 'POST']) # To render Homepage
def home_page():
    return render_template('index.html')


def sex_number(sex):
    if sex =='Male':
        return 1
    else:
        return 0 

def good_job(occupation):
    if occupation in ['Prof-specialty','Exec-managerial','Armed-Forces']:
        return 2
    elif occupation in ['Adm-clerical','Sales']:
        return 1
    else:
        return 0  

def educa_number(education):
    if education=='Preschool':
        return 1
    elif education=='1st-4th':
        return 2
    elif education=='5th-6th':
        return 3
    elif education=='7th-8th':
        return 4
    elif education=='9th':
        return 5
    elif education=='10th':
        return 6
    elif education=='11th':
        return 7
    elif education=='12th':
        return 8
    elif education=='HS-grad':
        return 9
    elif education=='Some-college':
        return 10
    elif education=='Assoc-voc':
        return 11
    elif education=='Assoc-acdm':
        return 12
    elif education=='Bachelors':
        return 13
    elif education=='Masters':
        return 14
    elif education=='Prof-school':
        return 15
    else:
        return 16              

          




@app.route('/salary', methods=['POST'])  # This will be called from UI
def math_operation():
    if (request.method=='POST'):
        #operation=request.form['operation']
        
        age =int(request.form['age'])
        education =(request.form['education'])
        occupation =request.form['occupation']
        sex = request.form['sex']

        education = int(educa_number(education))
        occupation = int(good_job(occupation))
        sex = int(sex_number(sex))

        
        int_features = []
        
        int_features.append(age)
        int_features.append(education)
        int_features.append(occupation)
        int_features.append(sex)

        

        final_features = [np.array(int_features)]

        print('final features',final_features)

        



        prediction = model.predict(final_features)

        print(prediction) 
        
        if prediction == 0:
            verdict = 'the person has less salary than 50k'
        else:
            verdict = 'the person has more salary than 50k'    



        

        
        return render_template('results.html',result=verdict)  


if __name__ == '__main__':
    app.run(debug=True)          