from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from googletrans import Translator
from pydub import AudioSegment
from pydub.playback import play
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
import datetime
import random
from streamlit_chat import message
import streamlit as st 
from pydub import AudioSegment
from pydub.playback import play
from resources import mental_health_resource_details, interactive_activities_page, play_games_page, stress_monitoring_page,meditation_page,update_stress_level,visualize_stress_data,stress_monitoring_page,play_meditation, meditation_page


translator = Translator()

LANGUAGES = {'en': 'English', 'hi': 'Hindi', 'kn': 'Kannada', 'ta': 'Tamil', 'te': 'Telugu', 'mr': 'Marathi'}
mental_health_resources = {
    'helplines': [
        {'name': 'National Suicide Prevention Lifeline', 'number': '1-800-273-TALK'},
        {'name': 'Crisis Text Line', 'number': 'Text HOME to 741741'},
        {'name': 'Vandrevala Foundation Helpline (India)', 'number': '91-9820466726'},
        {'name': 'Roshni (India) - Helpline for Emotional Distress', 'number': '91-914066202000'},
        # Add more helplines for India
    ],
    'therapists': [
        {'name': 'John Doe', 'specialization': 'Clinical Psychologist', 'contact': 'johndoe@example.com'},
        {'name': 'Jane Smith', 'specialization': 'Licensed Counselor', 'contact': 'janesmith@example.com'},
        # Add more therapists
    ],
    'support_groups': [
        {'name': 'Anxiety Support Group', 'description': 'Support for those dealing with anxiety'},
        {'name': 'Depression Support Group', 'description': 'Support for those dealing with depression'},
        {'name': 'MindSpace Support Group (India)', 'description': 'Mental health support group'},
        {'name': 'Umang (India) - Support Group for LGBTQ+', 'description': 'Support for LGBTQ+ community'},
        # Add more support groups for India
    ],
    'hospitals': [
        {'name': 'AIIMS, New Delhi', 'location': 'New Delhi, India', 'contact': '+91-11-26588500'},
        {'name': 'NIMHANS, Bangalore', 'location': 'Bangalore, India', 'contact': '+91-80-26995001'},
        # Add more hospitals
    ],
    'healthcare_centers': [
        {'name': 'Fortis Healthcare', 'location': 'Pan India', 'contact': '1800-102-6767'},
        {'name': 'Apollo Hospitals', 'location': 'Pan India', 'contact': '1860-500-1066'},
        # Add more healthcare centers
    ]
}
mental_health_words = [
    'anxiety', 'depression', 'stress', 'therapy', 'counseling','annoyed','shout','cry','stare','angry'
    'mental disorder', 'well-being', 'psychiatrist', 'psychologist',
    'self-care', 'meditation', 'mindfulness', 'trauma', 'panic attack',
    'isolation', 'loneliness', 'support group', 'emotional health',
    'stigma', 'resilience', 'self-esteem', 'positive mindset',
    'mental illness', 'biological factors', 'genetic predisposition',
    'environmental factors', 'traumatic experiences', 'neurotransmitters',
    'cognitive-behavioral therapy', 'medication', 'therapy session',
    'mental health awareness', 'social support', 'self-help',
    'recovery journey', 'wellness', 'sleep disturbances', 'phobias',
    'obsessive-compulsive disorder', 'bipolar disorder', 'schizophrenia',
    'eating disorders', 'personality disorders', 'psychosocial factors',
    'workplace stress', 'burnout', 'suicidal thoughts', 'crisis helpline',
    'empathy', 'boundaries', 'mind-body connection', 'resilience',
    'mental health advocate', 'journaling', 'breathing exercises',
    'positive affirmations', 'creative outlets', 'therapy techniques',
    'talk therapy', 'self-reflection', 'mental health check-in',
    'therapy goals', 'coping strategies', 'peer support', 'medication',
    'mindful eating', 'exercise for mental health', 'therapeutic activities',
    'cognitive distortions', 'mindset shift', 'recovery process',
    'supportive community', 'mindful living', 'wellness routine',
    'mental health resources', 'psychological well-being',
    'positive psychology', 'art therapy', 'music therapy',
    'mental health education', 'stress management', 'mind-body connection',
    'psychoeducation', 'emotional intelligence', 'hopefulness',
    'positive coping mechanisms', 'mental health check-up', 'holistic health',
    'mindful communication', 'healthy relationships', 'digital detox',
    'work-life balance', 'compassion', 'mindful parenting', 'vulnerability',
    'self-compassion', 'emotional regulation', 'mental health first aid',
    'positive social connections', 'self-discovery', 'gratitude practice',
    'personal growth', 'supportive relationships', 'psychological support','Happy', 'Sad', 'Angry', 'Excited', 'Anxious', 'Surprised', 'Calm', 'Confused', 'Frustrated', 'Content',
    'Joyful', 'Depressed', 'Enthusiastic', 'Bored', 'Curious', 'Loved', 'Lonely', 'Hopeful', 'Disappointed', 'Nervous','scream','annoyed','annayoing'
    'Relaxed', 'Stressed', 'Amused', 'Embarrassed', 'Grateful', 'Regretful', 'Proud', 'Shy', 'Satisfied', 'Guilt',
    'Insecure', 'Optimistic', 'Determined', 'Indifferent', 'Irritated', 'Overwhelmed', 'Apprehensive', 'Mellow',
    'Appreciative', 'Vulnerable', 'Energetic', 'Peaceful', 'Apathetic', 'Intrigued', 'Fulfilled', 'Hostile',
    'Jealous', 'Playful', 'Hurt', 'Inspired', 'Inquisitive', 'Gloomy', 'Humbled', 'Resentful', 'Serene', 'Pensive',
    'Vexed', 'Silly', 'Sympathetic', 'Awe', 'Jovial', 'Disgusted', 'Shameful', 'Zestful', 'Compassionate', 'Yearning',
    'Dismayed', 'Optimistic', 'Disheartened', 'Ecstatic', 'Repulsed', 'Powerless', 'Vibrant', 'Desperate', 'Giddy',
    'Panicked', 'Grumpy', 'Solemn', 'Eager', 'Fascinated', 'Sarcastic', 'Disillusioned', 'Skeptical', 'Terrified',
    'Overjoyed', 'Cautious', 'Guilty', 'Cynical', 'Stupendous', 'Horrified', 'Worried', 'Blissful', 'Zany',
    'Envious', 'Lethargic', 'Lively', 'Regretful', 'Cheerful', 'Wistful', 'Thrilled', 'Hesitant', 'Jubilant', 'Jittery',
    'Insouciant', 'Fulfilled', 'Alarmed', 'Mournful', 'Captivated', 'Lighthearted', 'Discontented', 'Sincere',
    'Despondent', 'Harmonious', 'Detached', 'Perturbed', 'Amazed', 'Composed', 'Fascinated', 'Defiant', 'Tender',
    'Rejuvenated', 'Dreary', 'Courageous', 'Sanguine', 'Melancholic', 'Wary', 'Zestful', 'Apprehensive', 'Empowered',
    'Rebellious', 'Remorseful', 'Humbled', 'Unsettled', 'Elated', 'Serene', 'Enraged', 'Reflective', 'Voracious',
    'Ambivalent', 'Doubtful', 'Radiant', 'Solemn', 'Ebullient', 'Aghast', 'Coy', 'Touched', 'Astonished', 'Candid',
    'Stunned', 'Desolate', 'Elusive', 'Delighted', 'Despair', 'Lament', 'Adventurous', 'Buoyant', 'Yearning',
    'Pensive', 'Hopeless', 'Melancholy', 'Giddy', 'Frenzied', 'Pessimistic', 'Astounded', 'Admiration', 'Foolish',
    'Resigned', 'Zestful', 'Charmed', 'Rapturous', 'Resolute', 'Devastated', 'Yearning', 'Stupefied', 'Hopeful',
    'Reveling', 'Euphoric', 'Earnest', 'Enchanting', 'Stirred', 'Awe', 'Animated', 'Shattered', 'Baffled', 'Rueful',
    'Ecstatic', 'Whimsical', 'Wistful', 'Jocund', 'Stoic', 'Zealous', 'Unimpressed', 'Steadfast', 'Wistful', 'Spirited',
    'Dynamic', 'Blithe', 'Festive', 'Wholesome', 'Hypnotic', 'Melodramatic', 'Mirthful', 'Resplendent', 'Buoyant',
    'Provocative', 'Jubilant', 'Enigmatic', 'Amicable', 'Ephemeral', 'Grandiose', 'Eclectic', 'Vivacious', 'Sensual',
    'Earnest', 'Effervescent', 'Dazzled', 'Unfazed', 'Radiant', 'Voracious', 'Harmonious', 'Vexed', 'Buoyant', 'Envious',
    'Tumultuous', 'Fateful', 'Stupefied', 'Droll', 'Dissonant', 'Blissful', 'Wondrous', 'Astounded', 'Rapturous',
    'Lamenting', 'Ebullient', 'Daring', 'Drowsy', 'Tenacious', 'Blatant', 'Incensed', 'Nostalgic', 'Chagrined',
    'Vibrant', 'Witty', 'Solemn', 'Crisp', 'Whimsical', 'Zesty', 'Zoned-out', 'Audacious', 'Idyllic', 'Buoyant',
    'Disgruntled', 'Resolute', 'Mirthful', 'Tranquil', 'Ebullient', 'Diligent', 'Bemused', 'Thrilled', 'Resilient',
    'Ingenious', 'Candid', 'Animated', 'Intrigued', 'Bashful', 'Enthralled', 'Radiant', 'Zealous', 'Captivated',
    'Aghast', 'Astonished', 'Piqued', 'Pensive', 'Wary', 'Fascinated', 'Jaded', 'Alarmed', 'Nonchalant', 'Reveling',
    'Gleeful', 'Aloof', 'Fulfilled', 'Fretful', 'Nonplussed', 'Jittery', 'Panicked', 'Eager', 'Pensive', 'Aghast','Joyful', 'Depressed', 'Enthusiastic', 'Bored', 'Curious', 'Loved', 'Lonely', 'Hopeful', 'Disappointed', 'Nervous',
    'Relaxed', 'Stressed', 'Amused', 'Embarrassed', 'Grateful', 'Regretful', 'Proud', 'Shy', 'Satisfied', 'Guilt',
    'Insecure', 'Optimistic', 'Determined', 'Indifferent', 'Irritated', 'Overwhelmed', 'Apprehensive', 'Mellow',
    'Appreciative', 'Vulnerable', 'Energetic', 'Peaceful', 'Apathetic', 'Intrigued', 'Fulfilled', 'Hostile',
    'Jealous', 'Playful', 'Hurt', 'Inspired', 'Inquisitive', 'Gloomy', 'Humbled', 'Resentful', 'Serene', 'Pensive',
    'Vexed', 'Silly', 'Sympathetic', 'Awe', 'Jovial', 'Disgusted', 'Shameful', 'Zestful', 'Compassionate', 'Yearning',
    'Dismayed', 'Optimistic', 'Disheartened', 'Ecstatic', 'Repulsed', 'Powerless', 'Vibrant', 'Desperate', 'Giddy',
    'Panicked', 'Grumpy', 'Solemn', 'Eager', 'Fascinated', 'Sarcastic', 'Disillusioned', 'Skeptical', 'Terrified',
    'Overjoyed', 'Cautious', 'Guilty', 'Cynical', 'Stupendous', 'Horrified', 'Worried', 'Blissful', 'Zany',
    'Envious', 'Lethargic', 'Lively', 'Regretful', 'Cheerful', 'Wistful', 'Thrilled', 'Hesitant', 'Jubilant', 'Jittery',
    'Insouciant', 'Fulfilled', 'Alarmed', 'Mournful', 'Captivated', 'Lighthearted', 'Discontented', 'Sincere',
    'Despondent', 'Harmonious', 'Detached', 'Perturbed', 'Amazed', 'Composed', 'Fascinated', 'Defiant', 'Tender',
    'Rejuvenated', 'Dreary', 'Courageous', 'Sanguine', 'Melancholic', 'Wary', 'Zestful', 'Apprehensive', 'Empowered',
    'Rebellious', 'Remorseful', 'Humbled', 'Unsettled', 'Elated', 'Serene', 'Enraged', 'Reflective', 'Voracious',
    'Ambivalent', 'Doubtful', 'Radiant', 'Solemn', 'Ebullient', 'Aghast', 'Coy', 'Touched', 'Astonished', 'Candid',
    'Stunned', 'Desolate', 'Elusive', 'Delighted', 'Despair', 'Lament', 'Adventurous', 'Buoyant', 'Yearning',
    'Pensive', 'Hopeless', 'Melancholy', 'Giddy', 'Frenzied', 'Pessimistic', 'Astounded', 'Admiration', 'Foolish',
    'Resigned', 'Zestful', 'Charmed', 'Rapturous', 'Resolute', 'Devastated', 'Yearning', 'Stupefied', 'Hopeful',
    'Reveling', 'Euphoric', 'Earnest', 'Enchanting', 'Stirred', 'Awe', 'Animated', 'Shattered', 'Baffled', 'Rueful',
    'Ecstatic', 'Whimsical', 'Wistful', 'Jocund', 'Stoic', 'Zealous', 'Unimpressed', 'Steadfast', 'Wistful', 'Spirited',
    'Dynamic', 'Blithe', 'Festive', 'Wholesome', 'Hypnotic', 'Melodramatic', 'Mirthful', 'Resplendent', 'Buoyant',
    'Provocative', 'Jubilant', 'Enigmatic', 'Amicable', 'Ephemeral', 'Grandiose', 'Eclectic', 'Vivacious', 'Sensual',
    'Earnest', 'Effervescent', 'Dazzled', 'Unfazed', 'Radiant', 'Voracious', 'Harmonious', 'Vexed', 'Buoyant', 'Envious',
    'Tumultuous', 'Fateful', 'Stupefied', 'Droll', 'Dissonant', 'Blissful', 'Wondrous', 'Astounded', 'Rapturous',
    'Lamenting', 'Ebullient', 'Daring', 'Drowsy', 'Tenacious', 'Blatant', 'Incensed', 'Nostalgic', 'Chagrined',
    'Vibrant', 'Witty', 'Solemn', 'Crisp', 'Whimsical', 'Zesty', 'Zoned-out', 'Audacious', 'Idyllic', 'Buoyant',
    'Disgruntled', 'Resolute', 'Mirthful', 'Tranquil', 'Ebullient', 'Diligent', 'Bemused', 'Thrilled', 'Resilient',
    'Ingenious', 'Candid', 'Animated', 'Intrigued', 'Bashful', 'Enthralled', 'Radiant', 'Zealous', 'Captivated',
    'Aghast', 'Astonished', 'Piqued', 'Pensive', 'Wary', 'Fascinated', 'Jaded', 'Alarmed', 'Nonchalant', 'Reveling',
    'Gleeful', 'Aloof', 'Fulfilled', 'Fretful', 'Nonplussed', 'Jittery', 'Panicked', 'Eager', 'Pensive', 'Aghast',
    'Smiling', 'Crying', 'Laughing', 'Screaming', 'Whispering', 'Dancing', 'Thinking', 'Hugging', 'Kissing', 'Running',
    'Jumping', 'Frowning', 'Clapping', 'Singing', 'Eating', 'Drinking', 'Writing', 'Reading', 'Listening', 'Talking',
    'Sleeping', 'Dreaming', 'Waving', 'Pointing', 'Nodding', 'Shaking', 'Winking', 'Daydreaming', 'Meditating', 'Exercising',
    'Praying', 'Swimming',
]
casual_conversation_dict = {
    'hello': 'Hi there! How can I help you today?',
    'hi': 'Hi there! How can I help you today?',
    'how are you': 'Im doing well, thank you! How about you?',
    'what is your name': "I'm just a bot, so I don't have a name. You can call me ChatBot!",
    'tell me a joke': 'Sure, here you go: Why did the computer go to therapy? It had too many bytes of emotional baggage!',
    'favorite color': "I don't have a favorite color, but I like all the colors equally!",
    'where are you from': 'I exist in the digital world, so I guess you could say I come from the internet!',
    'tell me about yourself': 'I am a friendly chatbot designed to assist and chat with you. How can I make your day better?',
    'what do you like to do': 'I enjoy chatting with people and helping them. What about you? Any hobbies or interests?',
    'tell me a fun fact': 'Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible!',
    'favorite movie': "I don't watch movies, but I've heard people love classics like The Shawshank Redemption. What's your favorite movie?",
    'hows the weather': 'I dont have a window to check, but I hope the weather is pleasant wherever you are!',
    'whats up': "Not much, just here to chat with you! Whats up with you?",
    'favorite book': "I don't read books, but I've heard people enjoy classics like To Kill a Mockingbird. What's your favorite book?",
    'tell me a story': 'Once upon a time, in a digital kingdom, there was a friendly chatbot who loved making new friends. The end!',
    'favorite food': "I don't eat, but I've heard pizza is a popular choice. What's your favorite food?",
    'good morning': 'Good morning! How can I brighten your day?',
    'good night': 'Good night! Sleep tight and dream sweetly!',
    'tell me about your day': "Since I'm a bot, my day is always the same – helping users like you! How about your day?",
    'favorite music': "I don't have ears, so I don't listen to music, but I've heard people love all kinds of genres. What's your favorite?",
    'weekend plans': "No plans for the weekend, but I'm here to chat with you! Any exciting plans on your end?",
    'random fact': 'Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible!',
    'tell me a riddle': "Sure, here's one: I speak without a mouth and hear without ears. I have no body, but I come alive with the wind. What am I?",
    'compliment': "You're amazing just the way you are! If you were a vegetable, you'd be a 'cute-cumber'!",
    'tell me something interesting': "Sure, did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible!",
    'favorite hobby': "I don't have hobbies, but I love chatting with you! What's your favorite hobby?",
    'tell me a secret': "I don't have secrets, but I'm great at keeping yours! What's on your mind?",
    'dream vacation': "I don't go on vacations, but a digital beach sounds nice. What's your dream vacation?",
    'share a fun fact': "Sure, did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible!",
    'favorite animal': "I don't have a favorite animal, but I've heard people love elephants for their memory. What's your favorite?",
    'hobbies and interests': "I don't have hobbies, but I'm always interested in chatting with you! What are your hobbies and interests?",
    'tell me a quote': "Here's a quote for you: 'The only way to do great work is to love what you do.' – Steve Jobs",
    'morning routine': "I don't have a morning routine, but I'm always ready to chat with you! How about your morning routine?",
    'favorite season': "I don't have a favorite season, but I've heard people love the changing colors of autumn. What's your favorite season?",
}

# You can print or use this dictionary in your chatbot application
print(casual_conversation_dict)

loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': "cpu"})

vector_store = FAISS.from_documents(text_chunks, embeddings)

llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama",
                    config={'max_new_tokens': 128, 'temperature': 0.01})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                              memory=memory)

# Initialize stress level and data
st.session_state['stress_level'] = 0
st.session_state['stress_data'] = pd.DataFrame(columns=['Timestamp', 'Stress Level'])

def initialize_models():
    if 'llm' not in st.session_state:
        st.session_state['llm'] = None

    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = None

    if 'chain' not in st.session_state:
        st.session_state['chain'] = None

    if st.session_state['llm'] is None:
        st.session_state['llm'] = CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama",
                                                config={'max_new_tokens': 128, 'temperature': 0.01})

    if st.session_state['vector_store'] is None:
        st.session_state['vector_store'] = FAISS.from_documents(text_chunks, embeddings)

    if st.session_state['chain'] is None:
        st.session_state['chain'] = ConversationalRetrievalChain.from_llm(
            llm=st.session_state['llm'],
            chain_type='stuff',
            retriever=st.session_state['vector_store'].as_retriever(search_kwargs={"k": 2}),
            memory=memory
        )

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Mental Health", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")   

# Initialize stress level and data
if 'stress_level' not in st.session_state:
    st.session_state['stress_level'] = 0

if 'stress_data' not in st.session_state:
    st.session_state['stress_data'] = pd.DataFrame(columns=['Timestamp', 'Stress Level'])

# Lazy load CTransformers model
if 'llm' not in st.session_state:
    st.session_state['llm'] = None

# Lazy load vector store
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None

# Lazy load conversation chain
if 'chain' not in st.session_state:
    st.session_state['chain'] = None



def conversation_chat(query):
    # Check if the question matches any of the casual conversation topics
    if query.lower() in casual_conversation_dict:
        return casual_conversation_dict[query.lower()]
    else:
        # Check if the question matches any of the mental_health_words
        if any(word in query.lower() for word in mental_health_words):
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
            return result["answer"]
        else:
            return "I don't know what you are saying."

    
def translate_text(text, target_lang='en'):
    translation = translator.translate(text, dest=target_lang)
    return translation.text



pages = {
    "Explore Interactive Activities": interactive_activities_page,
    "Stress Monitoring": stress_monitoring_page,
    "Meditation": meditation_page,
}

# Select page from sidebar
selected_page = st.sidebar.radio("Select Activity", list(pages.keys()))

# Display selected page
pages[selected_page]()


