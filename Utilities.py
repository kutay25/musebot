import csv, re
import streamlit as st



def normalize_string(s):
    # Lowercase and remove non-alphanumeric characters (excluding spaces)
    return re.sub(r'[^a-z0-9\s]', '', s.lower())

@st.cache_data
def appendVideo(generated_message, input_csv):
    input_csv = input_csv + ".csv"
    with open(input_csv, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        titles = [(row['title'], row['ytid']) for row in reader]

    normalized_input = normalize_string(generated_message)
    input_tokens = normalized_input.split()
    
    for title_ytid in titles:
        song_title = title_ytid[0]
        normalized_title = normalize_string(song_title)
        words_title = normalized_title.split()
        
        match_cnt = 0
        word_cnt = len(words_title)
        threshold = 0.75

        if(word_cnt == 0):
            print(f"word_cnt is equal to 0 for {title_ytid}")
            continue
        # If at least %75 of title's words are in the answer, then paste the youtube link.
        # Alternative: Check if two consecutive words are in the answer! Link to video if above %50 of title's pairs.
        for word in words_title:
            if(word in normalized_input):
                match_cnt += 1

        match_percentage = match_cnt / word_cnt
        if(match_percentage > threshold and title_ytid[1] not in st.session_state['videos']):
            print(f"Matched title at {song_title} with ratio {match_cnt / word_cnt} at input {generated_message}")
            st.video("https://youtube.com/watch?v="+title_ytid[1])
            st.session_state['videos'].append(title_ytid[1])
