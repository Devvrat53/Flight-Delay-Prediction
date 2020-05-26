from flask import Flask, request, jsonify, render_template, url_for , request
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# Import dataset 
df = pd.read_csv('Data/Processed_data15.csv')

# Label Encoding
le_carrier = LabelEncoder()
df['carrier'] = le_carrier.fit_transform(df['carrier'])

le_dest = LabelEncoder()
df['dest'] = le_dest.fit_transform(df['dest'])

le_origin = LabelEncoder()
df['origin'] = le_origin.fit_transform(df['origin'])

# Converting Pandas DataFrame into a Numpy array
X = df.iloc[:, 0:6].values # from column(years) to column(distance)
y = df['delayed']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25,random_state=61) # 75% training and 25% test

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    year = request.form['year']
    month = request.form['month']
    day = request.form['day']
    carrier = request.form['carrier']
    origin = request.form['origin']
    dest = request.form['dest']
    year = int(year)
    month = int(month)
    day = int(day)
    carrier = str(carrier)
    origin = str(origin)
    dest = str(dest)
    
    if year >= 2013:
        x1 = [year,month,day]
        x2 = [carrier, origin, dest]
        x1.extend(x2)
        df1 = pd.DataFrame(data = [x1], columns = ['year', 'month', 'date', 'carrier', 'origin', 'dest'])
        
        df1['carrier'] = le_carrier.transform(df1['carrier'])
        df1['origin'] = le_origin.transform(df1['origin'])
        df1['dest'] = le_dest.transform(df1['dest'])
        
        x = df1.iloc[:, :6].values
        ans = model.predict(x)
        output = ans
    
    return render_template('index.html', prediction_text=output)
    
if __name__ == '__main__':
	app.run(debug=False)
# For mac, make 'app.run(debug=True)'