
import streamlit as st
import pandas as pd
import joblib

# Load the preprocessor and classifier
preprocessor = joblib.load('preprocessor.joblib')
classifier = joblib.load('classifier.joblib')

st.title('DOTA2 GAME OUTCOME PREDICTION')
st.image('https://i.ytimg.com/vi/4KSNTN7IcR4/sddefault.jpg', caption='ACCEPT???', use_column_width=True)

# Define pre-saved data
pre_saved_data = {
    'kills': 10,
    'deaths': 2,
    'assists': 8,
    'gold_per_min': 500,
    'xp_per_min': 600,
    'level': 20,
    'duration_min': 45,
    'gametime': 'Evening',
    'gameplay_type': 'Burst'
}

def user_input_features():
    if st.button('Use Pre-saved Data'):
        kills = st.session_state.kills = pre_saved_data['kills']
        deaths = st.session_state.deaths = pre_saved_data['deaths']
        assists = st.session_state.assists = pre_saved_data['assists']
        gold_per_min = st.session_state.gold_per_min = pre_saved_data['gold_per_min']
        xp_per_min = st.session_state.xp_per_min = pre_saved_data['xp_per_min']
        level = st.session_state.level = pre_saved_data['level']
        duration_min = st.session_state.duration_min = pre_saved_data['duration_min']
        gametime = st.session_state.gametime = pre_saved_data['gametime']
        gameplay_type = st.session_state.gameplay_type = pre_saved_data['gameplay_type']
    else:
        kills = st.number_input('Enter number of kills', min_value=0, value=st.session_state.get('kills', 0))
        deaths = st.number_input('Enter number of deaths', min_value=0, value=st.session_state.get('deaths', 0))
        assists = st.number_input('Enter number of assists', min_value=0, value=st.session_state.get('assists', 0))
        gold_per_min = st.number_input('Enter gold earned per minute', min_value=0, value=st.session_state.get('gold_per_min', 0))
        xp_per_min = st.number_input('Enter XP earned per minute', min_value=0, value=st.session_state.get('xp_per_min', 0))
        level = st.number_input('Enter level reached', min_value=1, value=st.session_state.get('level', 1))
        duration_min = st.number_input('Enter game duration in minutes', min_value=1, value=st.session_state.get('duration_min', 1))
        gametime = st.selectbox('Select game time', ['Morning', 'Afternoon', 'Evening', 'Midnight'], index=['Morning', 'Afternoon', 'Evening', 'Midnight'].index(st.session_state.get('gametime', 'Morning')))
        gameplay_type = st.selectbox('Select gameplay type', ['Healing', 'Burst', 'Gank', 'Control', 'Damage'], index=['Healing', 'Burst', 'Gank', 'Control', 'Damage'].index(st.session_state.get('gameplay_type', 'Healing')))

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

st.subheader('User Input features')
st.write(input_df)

if st.button('Predict'):
    processed_features = preprocessor.transform(input_df)
    prediction = classifier.predict(processed_features)
    result = 'Win' if prediction[0] else 'Lose'
    st.subheader('Prediction')
    st.write(result)
