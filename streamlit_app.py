import streamlit as st
import pandas as pd
import joblib

# Load the preprocessor and classifier
preprocessor = joblib.load('preprocessor.joblib')
classifier = joblib.load('classifier.joblib')

#print('Model loaded Successfully')

# Title
st.title('DOTA2 GAME OUTCOME PREDICTION')

st.image('https://i.ytimg.com/vi/4KSNTN7IcR4/sddefault.jpg', caption='ACCEPT???', use_column_width=True)

# Collecting user input features
def user_input_features():
    kills = st.number_input('Enter number of kills', min_value=0, value=0)
    deaths = st.number_input('Enter number of deaths', min_value=0, value=0)
    assists = st.number_input('Enter number of assists', min_value=0, value=0)
    gold_per_min = st.number_input('Enter gold earned per minute', min_value=0, value=0)
    xp_per_min = st.number_input('Enter XP earned per minute', min_value=0, value=0)
    level = st.number_input('Enter level reached', min_value=1, value=1)
    duration_min = st.number_input('Enter game duration in minutes', min_value=1, value=1)
    gametime = st.selectbox('Select game time', ['Morning', 'Afternoon', 'Evening', 'Midnight'])
    gameplay_type = st.selectbox('Select gameplay type', ['Healing', 'Burst', 'Gank', 'Control', 'Damage'])
    data = {'kills': [kills],
            'deaths': [deaths],
            'assists': [assists],
            'gold_per_min': [gold_per_min],
            'xp_per_min': [xp_per_min],
            'level': [level],
            'duration_min': [duration_min],
            'gametime': [gametime],
            'gameplay_type': [gameplay_type]}
    features = pd.DataFrame(data)
    return features

input_df = user_input_features()



# Display the input dataframe
st.subheader('User Input features')
st.write(input_df)

# Preprocess the input data and predict
if st.button('Predict'):
    processed_features = preprocessor.transform(input_df)
    prediction = classifier.predict(processed_features)
    result = 'Win' if prediction[0] else 'Lose'
    st.subheader('Prediction')
    st.write(result)






# Load the model and preprocessor
#model = pickle.load(open('C:\\Users\\Moran\\anaconda3\\envs\\myenv\\General Assembly\\Capstone\\finalized_model.joblib', 'rb'))
#preprocessor = load('C:\\Users\\Moran\\anaconda3\\envs\\myenv\\General Assembly\\Capstone\\preprocessor.pkl')
