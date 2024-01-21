import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from googletrans import Translator
import pandas as pd
import random
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from pydub import AudioSegment
from pydub.playback import play
if 'stress_level' not in st.session_state:
    st.session_state['stress_level'] = 0

if 'stress_data' not in st.session_state:
    st.session_state['stress_data'] = pd.DataFrame(columns=['Timestamp', 'Stress Level'])


mental_health_resources = {
    'helplines': [
        {'name': 'National Suicide Prevention Lifeline', 'number': '1-800-273-TALK'},
        # ...
    ],
    'therapists': [
        {'name': 'John Doe', 'specialization': 'Clinical Psychologist', 'contact': 'johndoe@example.com'},
        # ...
    ],
    'support_groups': [
        {'name': 'Anxiety Support Group', 'description': 'Support for those dealing with anxiety'},
        # ...
    ],
    'hospitals': [
        {'name': 'AIIMS, New Delhi', 'location': 'New Delhi, India', 'contact': '+91-11-26588500'},
        # ...
    ],
    'healthcare_centers': [
        {'name': 'Fortis Healthcare', 'location': 'Pan India', 'contact': '1800-102-6767'},
        # ...
    ]
}

def mental_health_resource_details(category):
    resources = mental_health_resources(category)
    if resources:
        return random.choice(resources)
    else:
        return None
    

def interactive_activities_page():
    # Create a new page or modal for interactive activities
    st.title("Interactive Activities to Improve Mental Health 🎮")
    
    # Add suggestions, games, or interactive activities here
    st.write("Try these activities to relax and improve your mood:")
    
    # Deep breathing exercises
    st.write("1. Deep Breathing Exercises:")
    st.write("   Inhale deeply through your nose for a count of 4, hold your breath for 4 counts, and exhale slowly for 4 counts. Repeat several times.")
    
    # Guided meditation
    st.write("2. Guided Meditation:")
    st.write("   Use meditation apps like Headspace or Calm for guided meditation sessions. It can help calm your mind and reduce stress.")
    
    # Mindfulness games
    st.write("3. Mindfulness Games:")
    st.write("   Play simple mindfulness games like Sudoku, crosswords, or jigsaw puzzles. They engage your mind and promote focus.")
    
    # Positive affirmations
    st.write("4. Positive Affirmations:")
    st.write("   Practice positive self-talk. Repeat affirmations such as 'I am strong,' 'I am resilient,' and 'I can overcome challenges.'")

    # Creative expression
    st.write("5. Creative Expression:")
    st.write("   Engage in creative activities like drawing, painting, or writing. Expressing yourself creatively can be therapeutic.")

    # Physical activity
    st.write("6. Physical Activity:")
    st.write("   Incorporate regular physical exercise into your routine. It releases endorphins, which are natural mood boosters.")

    # Relaxing music
    st.write("7. Relaxing Music:")
    st.write("   Listen to calming music or nature sounds. Music has a powerful impact on emotions and can help you relax.")

    # Connect with loved ones
    st.write("8. Connect with Loved Ones:")
    st.write("   Spend time with friends or family, either in person or through video calls. Social connections are essential for mental well-being.")

    # Gratitude journaling
    st.write("9. Gratitude Journaling:")
    st.write("   Keep a gratitude journal. Write down things you're grateful for each day. Focusing on the positive can shift your mindset.")

    # Virtual adventures
    st.write("10. Virtual Adventures:")
    st.write("   Explore virtual tours or experiences. Many museums and landmarks offer online tours, allowing you to 'travel' from home.")

if st.button("Explore Interactive Activities"):
    interactive_activities_page()

def play_games_page():
    st.title("Play Games 🎲")
    st.write("Choose a game to play:")
    
    # Game options
    game_options = {
        "number guessing game": number_guessing_game,
        "rock paper scissors": rock_paper_scissors,
        "simple quiz": simple_quiz,
        "Emotion Charades": play_emotion_charades,
        "Virtual Nature Walk": play_virtual_nature_walk,
        "Breathing Exercise Simulator": play_breathing_simulator,
        # Add more games as needed
    }

    selected_game = st.selectbox("Select a Game:", list(game_options.keys()))

    # Button to play selected game
    if st.button("Play Game"):
        game_function = game_options[selected_game]
        game_function()


# Additional games

def number_guessing_game():
    st.title("Number Guessing Game")
    
    # Generate a random number between 1 and 100
    secret_number = random.randint(1, 100)
    
    # User input for guessing
    guess = st.number_input("Guess a number between 1 and 100", min_value=1, max_value=100, step=1)
    
    # Check the guess
    if st.button("Check Guess"):
        if guess == secret_number:
            st.success("Congratulations! You guessed the correct number.")
        elif guess < secret_number:
            st.warning("Try a higher number.")
        else:
            st.warning("Try a lower number.")

def rock_paper_scissors():
    st.header("Rock, Paper, Scissors Game")
    user_choice = st.radio("Choose your move:", ["Rock", "Paper", "Scissors"])
    play_button = st.button("Play")

    if play_button:
        choices = ["Rock", "Paper", "Scissors"]
        computer_choice = random.choice(choices)

        st.write(f"You chose: {user_choice}")
        st.write(f"Computer chose: {computer_choice}")

        if user_choice == computer_choice:
            st.success("It's a tie!")
        elif (
            (user_choice == "Rock" and computer_choice == "Scissors") or
            (user_choice == "Paper" and computer_choice == "Rock") or
            (user_choice == "Scissors" and computer_choice == "Paper")
        ):
            st.success("You win!")
        else:
            st.error("Computer wins!")

def simple_quiz():
    st.header("Simple Quiz Game")
    question = "What is the capital of France?"
    options = ["Paris", "London", "Berlin", "Madrid"]
    user_answer = st.radio(question, options)
    submit_button = st.button("Submit Answer")

    if submit_button:
        correct_answer = "Paris"
        if user_answer == correct_answer:
            st.success("Correct! Paris is the capital of France.")
        else:
            st.error(f"Wrong answer. The correct answer is {correct_answer}")

def play_emotion_charades():
    st.write("Emotion Charades 😊😢😠")
    st.write("Act out or describe an emotion without using words. See if others can guess the emotion.")

    # Input box for describing emotion
    user_input = st.text_input("Describe an Emotion:")

    if user_input:
        st.write(f"Great description! Let others guess the emotion.")

def play_virtual_nature_walk():
    st.write("Virtual Nature Walk 🌳🚶‍♂️")
    st.write("Immerse yourself in a virtual nature walk. Describe the sounds, sights, and smells you experience.")

    # Input box for describing nature
    user_input = st.text_input("Describe Your Virtual Nature Walk:")

    if user_input:
        st.write(f"Close your eyes and imagine the serene nature walk you described.")

def play_breathing_simulator():
    st.write("Breathing Exercise Simulator 🌬️")
    st.write("Practice deep breathing exercises for relaxation. Inhale, hold, exhale.")

    # Button to simulate breaths
    if st.button("Inhale"):
        st.write("Take a deep breath in...")

    if st.button("Hold"):
        st.write("Hold your breath...")

    if st.button("Exhale"):
        st.write("Exhale slowly...")

def update_stress_level(new_stress_level):
    timestamp = datetime.datetime.now()
    data = {'Timestamp': [timestamp], 'Stress Level': [new_stress_level]}

    # Initialize 'stress_data' if not yet initialized
    if 'stress_data' not in st.session_state:
        st.session_state['stress_data'] = pd.DataFrame(columns=['Timestamp', 'Stress Level'])

    # Append data to the stress_data DataFrame
    df = pd.DataFrame(data)
    st.session_state['stress_data'] = pd.concat([st.session_state['stress_data'], df], ignore_index=True)
# Function to visualize stress data
def visualize_stress_data():
    st.subheader("Stress Level Over Time")
    
    if not st.session_state['stress_data'].empty:
        # Convert 'Timestamp' to datetime object
        st.session_state['stress_data']['Timestamp'] = pd.to_datetime(st.session_state['stress_data']['Timestamp'])

        # Plot stress data
        fig, ax = plt.subplots(figsize=(10, 6))
        st.session_state['stress_data'].plot(x='Timestamp', y='Stress Level', marker='o', linestyle='-', ax=ax)
        plt.xlabel('Timestamp')
        plt.ylabel('Stress Level')
        plt.title('Stress Monitoring')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.info("No stress data available. Monitor your stress level to see the visualization.")

def stress_monitoring_page():
    st.title("Stress Monitoring 📊")

    # Stress level slider
    stress_level = st.slider("Current Stress Level (0-10):", min_value=0, max_value=10, step=1)

    # Button to update stress level
    if st.button("Update Stress Level"):
        update_stress_level(stress_level)
        st.success(f"Stress level updated to {stress_level}.")

    # Visualize stress data
    visualize_stress_data()


# Display stress monitoring page
stress_monitoring_page()


def play_meditation(audio_file_path):
    st.audio(audio_file_path, format='audio/mp3', start_time=0)

# Meditation page
def meditation_page():
    st.title("Meditation 🧘‍♂️")

    st.write("Take a moment to relax and practice meditation. Choose a guided meditation session below:")

    # Guided meditation sessions
    meditation_sessions = {
        'Session 1': 'meditation.mp3',
        'Session 2': 'meditation.mp3',
        # Add more sessions as needed
    }

    selected_session = st.selectbox("Select a Guided Meditation Session:", list(meditation_sessions.keys()))

    # Button to play selected meditation session
    if st.button("Play Meditation Session"):
        audio_file_path = meditation_sessions[selected_session]
        play_meditation(audio_file_path)

# Display the meditation page
meditation_page()

