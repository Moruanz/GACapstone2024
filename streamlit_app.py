import streamlit as st
import pandas as pd
import joblib

# Load the preprocessor and classifier
preprocessor = joblib.load('preprocessor.joblib')
classifier = joblib.load('classifier.joblib')

st.title('DOTA2 GAME OUTCOME PREDICTION')
st.image('https://i.ytimg.com/vi/4KSNTN7IcR4/sddefault.jpg', caption='ACCEPT???', use_column_width=True)

# Define two sets of pre-saved data
pre_saved_data_1 = {
    'kills': 4,
    'deaths': 7,
    'assists': 12,
    'gold_per_min': 240,
    'xp_per_min': 301,
    'level': 12,
    'duration_min': 30,
    'gametime': 'Afternoon',
    'gameplay_type': 'Gank'
}

pre_saved_data_2 = {
    'kills': 2,
    'deaths': 1,
    'assists': 12,
    'gold_per_min': 271,
    'xp_per_min': 487,
    'level': 19,
    'duration_min': 43,
    'gametime': 'Evening',
    'gameplay_type': 'Healing'
}

# Buttons to load pre-saved data
col1, col2 = st.columns(2)
with col1:
    if st.button('Use Demo Data 1'):
        st.session_state['data'] = pre_saved_data_1
        st.session_state['data_loaded'] = True
with col2:
    if st.button('Use Demo Data 2'):
        st.session_state['data'] = pre_saved_data_2
        st.session_state['data_loaded'] = True

def user_input_features():
    if st.session_state.get('data_loaded', False):
        data = st.session_state['data']
    else:
        data = {
            'kills': st.number_input('Enter number of kills', min_value=0, value=0),
            'deaths': st.number_input('Enter number of deaths', min_value=0, value=0),
            'assists': st.number_input('Enter number of assists', min_value=0, value=0),
            'gold_per_min': st.number_input('Enter gold earned per minute', min_value=0, value=0),
            'xp_per_min': st.number_input('Enter XP earned per minute', min_value=0, value=0),
            'level': st.number_input('Enter level reached', min_value=1, max_value = 30, value=1),
            'duration_min': st.number_input('Enter game duration in minutes', min_value=1,max_value = 368, value=1),
            'gametime': st.selectbox('Select game time', ['Morning', 'Afternoon', 'Evening', 'Midnight']),
            'gameplay_type': st.selectbox('Select gameplay type', ['Healing', 'Burst', 'Gank', 'Control', 'Damage'])
        }

    features = pd.DataFrame({k: [v] for k, v in data.items()})
    return features

input_df = user_input_features()

st.subheader('User Input features')
st.write(input_df)

# Preprocess the input data and predict
if st.button('Predict'):
    processed_features = preprocessor.transform(input_df)
    prediction = classifier.predict(processed_features)
    result = 'Win' if prediction[0] else 'Lose'
    st.subheader('Prediction')
    st.write(result)
