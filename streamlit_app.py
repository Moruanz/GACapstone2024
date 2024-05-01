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

# Function to reset session state
def reset_user_input():
    for key in pre_saved_data_1.keys():
        st.session_state[key] = 0
    st.session_state['level'] = 1
    st.session_state['duration_min'] = 1
    st.session_state['gametime'] = 'Morning'
    st.session_state['gameplay_type'] = 'Healing'
    st.session_state['data_loaded'] = False

# Buttons to load pre-saved data
col1, col2, col3 = st.columns(3)
with col1:
    if st.button('Use Demo Data 1'):
        st.session_state.update(pre_saved_data_1)
        st.session_state['data_loaded'] = True
with col2:
    if st.button('Use Demo Data 2'):
        st.session_state.update(pre_saved_data_2)
        st.session_state['data_loaded'] = True
with col3:
    if st.button('Reset Inputs'):
        reset_user_input()

def user_input_features():
    data = {
        'kills': st.number_input('Enter number of kills', min_value=0, value=st.session_state.get('kills', 0)),
        'deaths': st.number_input('Enter number of deaths', min_value=0, value=st.session_state.get('deaths', 0)),
        'assists': st.number_input('Enter number of assists', min_value=0, value=st.session_state.get('assists', 0)),
        'gold_per_min': st.number_input('Enter gold earned per minute', min_value=0, value=st.session_state.get('gold_per_min', 0)),
        'xp_per_min': st.number_input('Enter XP earned per minute', min_value=0, value=st.session_state.get('xp_per_min', 0)),
        'level': st.number_input('Enter level reached', min_value=1, max_value=30, value=st.session_state.get('level', 1)),
        'duration_min': st.number_input('Enter game duration in minutes', min_value=1, max_value=360, value=st.session_state.get('duration_min', 1)),
        'gametime': st.selectbox('Select game time', ['Morning', 'Afternoon', 'Evening', 'Midnight'], index=['Morning', 'Afternoon', 'Evening', 'Midnight'].index(st.session_state.get('gametime', 'Morning'))),
        'gameplay_type': st.selectbox('Select gameplay type', ['Healing', 'Burst', 'Gank', 'Control', 'Damage'], index=['Healing', 'Burst', 'Gank', 'Control', 'Damage'].index(st.session_state.get('gameplay_type', 'Healing')))
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
