import random
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
client = OpenAI()


# Generate a random number between 0 and 1
random_number = random.random()

def generate(related_memory, context, key_summary, user_name):

    prompt = f" You will be given the conversation history and the relevant memory, \
                you are able to use those information effectively to generate a response. \
                 Here is the summary of the conversation you had in the past with {user_name}: {key_summary}. \
                 Here is the recent utterances between you (LUFY) and {user_name}: {context}, \
                 and the memory most relevant to the {user_name}'s current utterance is: {related_memory}. \
                Remember to make the response casual like a real friend. Make the response in less than 50 words. "

    if random_number < 0.3:
        prompt = f"Now you will play the role of AI friend for user {user_name}." + prompt
    
    else:
        prompt = f"Now you will play the role of AI friend for a user." + prompt

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

    message = [{"role": "system", "content": "You are an AI friend, LUFY."},
                {"role": "user", "content": prompt}
            
            ]
    response = client.chat.completions.create(model="gpt-4o", messages=message)
    response = response.choices[0].message.content
    return response

