import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from chatbot import (
    display_chat_history,
    initialize_models,
    update_stress_level,
    visualize_stress_data,
    interactive_activities_page,
    play_games_page,
    stress_monitoring_page,
    meditation_page,
)
from googletrans import Translator
import pandas as pd
from streamlit_chat import message
# Function to initialize session state
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]
        
def visualize_stress_data():
    # Generate example stress data
    timestamps = pd.date_range('2022-01-01', periods=10, freq='D')
    stress_levels = np.random.randint(1, 10, size=(10,))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Create a DataFrame
    stress_df = pd.DataFrame({'Timestamp': timestamps, 'Stress Level': stress_levels})
    
    # Plot stress data
    plt.figure(figsize=(10, 6))
    plt.plot(stress_df['Timestamp'], stress_df['Stress Level'], marker='o')
    plt.title('Stress Data Visualization')
    plt.xlabel('Timestamp')
    plt.ylabel('Stress Level')
    plt.grid(True)

    # Display the plot using Streamlit
    st.pyplot()

    # Display the raw data table
    st.write("Raw Stress Data:")
    st.dataframe(stress_df)


def handle_additional_functions():
    st.header("Explore Additional Functions")
    st.write("Use the following buttons to explore additional functionalities of the HealthCare ChatBot:")

    # Button to explore interactive activities
    if st.button("Explore Interactive Activities", key="explore_activities"):
        interactive_activities_page()
        update_stress_level(1)  # Increment stress level after using the function

    # Increment stress level after using the function

    # Button to monitor stress
    if st.button("Monitor Stress", key="monitor_stress"):
        stress_monitoring_page()
        update_stress_level(1)  # Increment stress level after using the function

    # Button to practice meditation
    if st.button("Practice Meditation", key="practice_meditation"):
        meditation_page()
        update_stress_level(3)  # Increment stress level after using the function

def main():
    st.title("HealthCare ChatBot 🧑🏽‍⚕️")
    initialize_session_state()
    display_chat_history()
    initialize_models()
    # Menu options
    menu_options = ["Explore Interactive Activities", "Monitor Stress", "Practice Meditation", "Visualize Stress Data"]
    selected_menu_option = st.sidebar.selectbox("Select an option:", menu_options)

    # Display content based on the selected menu option
    if selected_menu_option == "Explore Interactive Activities":
        update_stress_level(1)
    elif selected_menu_option == "Monitor Stress":
        stress_monitoring_page()
        update_stress_level(1)
    elif selected_menu_option == "Practice Meditation":
        meditation_page()
        update_stress_level(3)
    elif selected_menu_option == "Visualize Stress Data":
        visualize_stress_data()

if __name__ == "__main__":
    main()