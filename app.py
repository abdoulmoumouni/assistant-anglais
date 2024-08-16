
import streamlit as st
import os
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import wordnet
from nltk.corpus import cmudict
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
#from st_audiorec import st_audiorec



st.set_page_config(
    page_title="Definitions",
    page_icon=":book:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.streamlit.io/help",
        "Report a bug": "https://github.com/streamlit/streamlit/issues",
        "About": "# This is a header. This is an *extremely* cool app!"
    }
)

# Use any translator you like, in this example GoogleTranslator
def translate_to_french(word):
    translated = GoogleTranslator(source='auto', target='fr', max_retries=10).translate(word)  # output -> Weiter so, du bist großartig
    return translated

def load_course_data(file):
    return file.read().decode("utf-8")

def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, 'wb') as out_file:
        out_file.write(uploaded_file.getbuffer())
def translate_text(text, target_language):
    if target_language == "French":
        translated = GoogleTranslator(source='auto', target='fr', max_retries=10).translate(text)
    elif target_language == "Hausa":
        translated = GoogleTranslator(source='auto', target='ha', max_retries=10).translate(text)
    else:
        translated = text
    return translated

def translation(text, target_language):
    try:
        if target_language == "French":
            translated = GoogleTranslator(source='auto', target='fr', max_retries=10).translate(text)
        elif target_language == "Hausa":
            translated = GoogleTranslator(source='auto', target='ha', max_retries=10).translate(text)
        else:
            translated = text
        return translated
    except Exception as e:
        # If an exception occurs, return a custom error message
        return f"Error: {e}"

d = cmudict.dict()

def count_syllables(word):
    try:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except KeyError:
        # if word not found in cmudict
        return syllables(word)

def syllables(word):
    # fallback to rule-based syllable counting
    count = 0
    vowels = 'aeiouy'
    word = word.lower()
    if word[0] in vowels:
        count +=1
    for index in range(1,len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count +=1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count+=1
    if count == 0:
        count +=1
    return count

def comprehension_module(file_content):
    # Add a search input field
    search_word = st.text_input("Search for a word:")

    # Tokenize the file content
    tokens = nltk.word_tokenize(file_content)

    # Identify difficult words
    difficult_words = [word for word in tokens if count_syllables(word) > 2]
    if search_word:
        # Check if the search word is in the difficult words list
        if search_word in difficult_words:
            st.write(f"## {search_word}")

            # Get the word definitions
            synsets = wordnet.synsets(search_word)

            if synsets:
                for synset in synsets:
                    st.write("Definition:", synset.definition())

                    # Get the synonyms
                    synonyms = synset.lemmas()
                    synonyms_list = [lemma.name() for lemma in synonyms if lemma.name() != search_word]

                    if synonyms_list:
                        st.write("Synonyms:", ", ".join(synonyms_list))

                    # Get the examples of usage
                    examples = synset.examples()

                    if examples:
                        st.write("Examples:")
                        for example in examples:
                            st.write("-", example)

                    st.write("---")
        else:
            st.write("Word not found in the difficult words list.")
    else:
        # Ensure uniqueness before display
        difficult_words = list(set(difficult_words)) 
        st.write(f"{difficult_words }")

        st.write("Difficult Words:")
        for word in difficult_words:
            st.write(f"### {word}")

            # Get the word definitions (and handle potential duplicates here)
            synsets = wordnet.synsets(word) 
            if synsets:
                for synset in synsets:
                    st.write("Definition:", synset.definition())

                    synonyms = synset.lemmas()
                    synonyms_list = [lemma.name() for lemma in synonyms if lemma.name() != word]

                    if synonyms_list:
                        st.write("Synonyms:", ", ".join(synonyms_list))

                    examples = synset.examples() 
                    if examples:
                        st.write("Examples:")
                        for example in examples:
                            st.write("-", example)

            st.write("---")




def text_summary(file_content):
    # Tokenize the text into individual words
    tokens = word_tokenize(file_content)

    # Remove stopwords from the tokens
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [t for t in tokens if t not in stop_words]

    # Use a simple summarization algorithm to extract key sentences
    sentences = sent_tokenize(file_content)
    summary_sentences = []
    for sentence in sentences:
        score = 0
        words = word_tokenize(sentence)
        for word in words:
            if word in filtered_tokens:
                score += 1
        summary_sentences.append((sentence, score))
    summary_sentences.sort(key=lambda x: x[1], reverse=True)

    # Return the top N sentences as a summary
    return [s[0] for s in summary_sentences[:10]]
        
def vocabulaire_module():
    st.write("### Vocabulaire")

    # Get the sentence
    sentence = st.text_input("Enter a sentence:")

    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence)

    # Display the tokens
    st.write("Tokens:", tokens)

    # Display the word details
    if tokens:
        st.subheader("Word Details")

        col1, col2 = st.columns([4, 2])

        for i, word in enumerate(tokens):
            with col1:
                st.write(f"**{word}**")

                # Get the word definitions
                synsets = wordnet.synsets(word)

                if synsets:
                    for synset in synsets:
                        st.write("Definition:", synset.definition())

                        # Get the synonyms
                        synonyms = synset.lemmas()
                        synonyms_list = [lemma.name() for lemma in synonyms if lemma.name() != word]

                        if synonyms_list:
                            st.write("Synonyms:", ", ".join(synonyms_list))

                        # Get the examples of usage
                        examples = synset.examples()

                        if examples:
                            st.write("Examples:")
                            for example in examples:
                                st.write("-", example)

                        st.write("---")

                with col2:
                    st.write("Translation Options:")
                    translation_language = st.radio(f"Translate to (Word {i+1})", options=["French", "Hausa"], key=f"translation_{i}")

                    if st.button("Translate"):
                        translated_definition = translate_text(synset.definition(), translation_language)
                        st.write("Translated Definition:", translated_definition)

                    st.write("---")

import playsound  # You might need to install it: pip install playsound
from gtts import gTTS
def practice_listening2(file_content):
    """
    Presents audio content from the file for listening practice.

    Args:
        file_content (str): The text content of the audio file or lesson.
    """
    with st.expander("Listening Practice"):
        st.write(f"Listen to the following passage:") 

        #  2. Play the Audio
        # Create the audio file when the function is called
        tts = gTTS(text=file_content, lang='en') # Change 'en' for other languages
        audio_file = "temp_audio.mp3"
        tts.save(audio_file)

        # Use Streamlit's audio method to play the audio directly in the app
        st.audio(audio_file, format='audio/mp3')

        #  3. Add More Features (e.g., comprehension questions)
        st.write("----")


def assistant_anglais(file_content):
    tabs = st.tabs(["Translation","Vocabulaire","Compréhension", "Questions-réponses","Resumé", "Guided composition"])

    with tabs[0]:
        st.write("### Translation")
        # Create a title for the app
        st.title("Language Translator")

        # Create input box for user to enter text
        text_input = st.text_area("Enter text to translate:", height=200)

        # Create dropdown menu for target language
        target_language = st.selectbox(
            "Select target language:",
            ["English (Default)", "French", "Hausa"]
        )

        # Add a button to trigger translation
        if st.button("Translate "):
            try:
                # Use the translate_text function with user's input and selected language
                translated_text = translation(text_input, {
                    "English (Default)": "",
                    "French": "French",
                    "Hausa": "Hausa"
                }.get(target_language, ""))
                
                # Display the translated text
                st.write("Translated Text:")
                st.write(translated_text)
            except ConnectionError:
                # If a connection error occurs, display a custom error message
                st.error("Error: Internet connection failed. Please check your internet connection and try again.")
        
        
    with tabs[1]:
        st.write("### vocabulaire")
        vocabulaire_module()

    with tabs[2]:
        st.write("### Compréhension")
        comprehension_module(file_content)
    
    with tabs[3]:
        st.write("### Quiz")

        # Tokenize the file content into sentences
        sentences = nltk.sent_tokenize(file_content)

        # Select a random sentence for the quiz
        quiz_sentence = random.choice(sentences)

        # Remove a random word from the sentence to create a question
        words = nltk.word_tokenize(quiz_sentence)
        word_freq = nltk.FreqDist(words)
        difficult_words = [word for word in words if word_freq[word] <= 2]  

        if difficult_words:
            correct_answer = random.choice(difficult_words)  
            blank_word = correct_answer  # Use the same variable name as before
            correct_index = [i for i, x in enumerate(words) if x == 
        correct_answer][0]  # Find the index of the correct answer
            words[correct_index] = "______"
            quiz_question = " ".join(words)

            # Display the quiz question
            st.write(f"Question: {quiz_question}")

            # Create a text input field for the user to answer
            user_answer = st.text_input("Enter your answer:", key="quiz_answer")

            # Add a submit button
            submit_button = st.button(label="Submit", key="submit_button")

            if submit_button:
                # Check if the user's answer is correct
                if user_answer.strip().lower() == blank_word.lower():
                    st.write("Correct!")
                else:
                    st.write(f"Sorry, the correct answer is {blank_word}.")
        else:
            st.write("No difficult words found in the text.")
                
    with tabs[4]:
        # Summarize the text
        summary = text_summary(file_content)

        # Display the summary
        st.subheader("Summary:")
        for sentence in summary:
            st.write(sentence)
    with tabs[5]:
       
        tokens = word_tokenize(file_content)
        stop_words = set(nltk.corpus.stopwords.words('english'))
        tokens_stop = [word for word in tokens if not word in stop_words]

        # Identifier les mots-clés (keywords) dans le texte
        keywords = [token for token in tokens_stop if token.isalpha()]
        mots_cles = []
        for mot in keywords:
            if len(mot) > 2 and mot.lower() not in ["the", "a", "an"]:
                mots_cles.append(mot)

        # Utiliser les mots-clés pour déterminer le thème
        theme = ""
        if len(mots_cles) > 0:
            theme = ", ".join(mots_cles[:3])
        else:
            theme = "Inconnu"

        # Demander à l'utilisateur d'écrire une composition guidée en anglais
        st.write(f"Écrivez votre composition guidée sur le thème : {theme}")
        texte = st.text_area("Écrivez votre réponse", key="réponse")

        # Create a submit button
        submitted = st.button("Soumettre")

        if submitted:
            # Analyser le vocabulaire et la cohérence du texte écrit par 
            if texte:
                tokens_user = word_tokenize(texte)
                stop_words = set(nltk.corpus.stopwords.words('english'))
                tokens_stop_user = [word for word in tokens_user if not word in stop_words]
                
                # Identify incorrect words
                incorrect_words = [word for word in tokens_user if word.lower() not in tokens_stop_user]
                
                # Calculate the number of correct words
                correct_words = len(tokens_stop_user)
                
                # Calculate the number of sentences
                sentences = nltk.sent_tokenize(texte)
                num_sentences = len(sentences)
                
                # Calculate the average sentence length
                avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / num_sentences
                
                # Calculate the frequency of each word
                word_freq = {}
                for word in tokens_stop_user:
                    if word in word_freq:
                        word_freq[word] += 1
                    else:
                        word_freq[word] = 1
                
                # Identify the most common words
                most_common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                
                st.write(f"### Nombre de phrases : {num_sentences}")
                st.write(f"### Longueur moyenne des phrases : {avg_sentence_length:.2f} mots")
                st.write("### Mots les plus fréquents :")
                for word, freq in most_common_words:
                    if len(word)>1:
                        st.write(f"{word}: {freq} fois")
    # with tabs[6]:
    #     practice_listening2(file_content)
        
    #     wav_audio_data = st_audiorec()

    #     if wav_audio_data is not None:
    #         st.audio(wav_audio_data, format='audio/wav')
        

        

def main():
    st.title("Study App for Students")

    courses_dir = "data/courses"
    if not os.path.exists(courses_dir):
        os.makedirs(courses_dir)

    # Sidebar for navigation and file upload
    st.sidebar.title("Courses")
    uploaded_file = st.sidebar.file_uploader("Upload a Course", type=["txt"])

    if uploaded_file:
        course_name = uploaded_file.name.split('.')[0]
        save_path = os.path.join(courses_dir, uploaded_file.name)

        save_uploaded_file(uploaded_file, save_path)
        st.sidebar.success(f"Uploaded {uploaded_file.name} successfully!")

    # Read courses from the directory
    course_names = [f.split('.')[0] for f in os.listdir(courses_dir)]
    selected_course = st.sidebar.selectbox("Select a Course", course_names)

    if selected_course:
        course_file_path = os.path.join(courses_dir, selected_course + ".txt")
        with open(course_file_path, 'r') as file:
            file_content = file.read()

        st.header(f"{selected_course.title()} Lesson")
        with st.expander(f"{selected_course.title()} Lesson"):
            st.write(file_content)
    
    assistant_anglais(file_content)



if __name__ == "__main__":
    main()

