import random
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
client = OpenAI()


# Generate a random number between 0 and 1
random_number = random.random()

def generate_first_conversation(related_memory, key_summary, user_name,session):
    current_session_conversation_path = f"../memories/{user_name}_memories/conversation"

    prompt = f"You are starting up a conversation with a friend {user_name} whom you met a couple of times.\
                 Here is the most important memory from the past conversations: {related_memory}. \
                Here is the summary of the conversation you had in the past with {user_name}: {key_summary}. \
                 Generate an ice breaker to start a conversation. \
                Remember to make the response casual like a real friend. Make the response in less than 50 words. "

    if session > 4:
        directory = current_session_conversation_path

        # Function to extract the first two sentences from a file
        def extract_first_two_sentences(file_path):
            with open(file_path, 'r') as file:
                content = file.read()
                sentences = content.split('\n')[:1]
                return ' '.join(sentences) + '.'

        # Loop through each file and extract the first two sentences

        openings = ""

        for i in range(session-3, session):
            file_name = f"{i}.txt"
            file_path = os.path.join(directory, file_name)
            first_two_sentences = extract_first_two_sentences(file_path)
            openings = openings + ", " f"Here is the first 2 utterances from session {i}" + first_two_sentences 

        prompt = prompt + "Here are some of the openings from the most recent 3 sessions." + openings


    # Check if the random number is within the 10% probability range
    if random_number < 0.3:
        current_datetime = datetime.now()
        datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        time_text = f"By the way, it is {datetime_string} now."

        # If yes, concatenate the additional text
        prompt = prompt + " " + time_text
    else:
        # If no, keep the base text unchanged
        prompt = prompt
    
    message = [
                {
                    "role": "system",
                    "content": prompt
                }
            ]
    response = client.chat.completions.create(model="gpt-4o", messages=message)
    response = response.choices[0].message.content
    return response

