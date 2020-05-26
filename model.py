# Import libraries 
import pandas as pd 
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function

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
y = df['delayed'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=18) # 70% training and 30% test
# For 75% Train and 25% test use Random state =809

#Create a Random Forest Classifier
clf = RandomForestClassifier(random_state=18)
clf.fit(X_train, y_train)

# Saving model to disk
pickle.dump(clf,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
