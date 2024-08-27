#summarize a conversation at a given session for a given user
import json
from llama_index.core import StorageContext, load_index_from_storage, Document
import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()

def summarize_wisely(user_name,session):
    text_file_path = f'./memories/{user_name}_memories/conversation/{session}.txt'

    with open(text_file_path, 'r') as text_file:
        # Read the entire contents of the file into a string
        file_contents = text_file.read()

    def generate(prompt, content):
        message = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{prompt}"},
                    {"role": "system", "content": "Sure, I will do my best to assist you."},
                    {"role": "user", "content": f"{content}"}]
        response = client.chat.completions.create(model="gpt-4o", messages=message)
        response = response.choices[0].message.content
        return response

    #below are the code for summarizing the summaries

    def summarize_daily_conversation(content):
        prompt = 'Please summarize the following dialogue as concisely as possible, extracting the main themes and key information. \
            Do not just list the importance utterances, make a concise summary. \
            If there are multiple key events, you may summarize them separately. \
            If the user mentions about their profile information such as age, gender, occupation and other type of personal information, \
            make sure to include that into the summary. \
                Make sure to include friends or family members mentioned in the dialogue.\
                Make the summary in less than 50 words.'
        return generate(prompt,content)

    summarized_text = summarize_daily_conversation(file_contents)

    filepath1 = f'./memories/{user_name}_memories/docstore.json'

    # Load the JSON data from your file
    with open(filepath1, "r") as file:
        data1 = json.load(file)

    importance = -1
    perplexity = 160
    arousal = -1
    memory_strength = -1


    #get the appropriate IPE
    # Iterate through the data and update "IPE" 
    for key, entry in data1["docstore/data"].items():

        first_mentioned_session = entry["__data__"]["metadata"]["first_mentioned_session"]

        if first_mentioned_session == session:

            importance1 = entry["__data__"]["metadata"]["importance"]
            perplexity1 = entry["__data__"]["metadata"]["perplexity"]
            arousal1 = entry["__data__"]["metadata"]["arousal"]
            memory_strength1 = entry["__data__"]["metadata"]["memory_strength"]

            if importance1 > importance:
                importance = importance1
            if perplexity1 < perplexity:
                perplexity = perplexity1
            if arousal1 > arousal:
                arousal = arousal1
            if memory_strength1 > memory_strength:
                memory_strength = memory_strength1


    document = Document(
        text= summarized_text,
        metadata={
            'filetype': 'summary',
            'memory_strength':memory_strength,
            'importance':importance,
            'last_recalled_session':session,
            'first_mentioned_session':session,
            'valence':0,
            'arousal':arousal,
            'perplexity':perplexity,
            'unused_count':0
        }
    )

    # saving summarized conversation to disk
    storage_context = StorageContext.from_defaults(persist_dir=f'./memories/{user_name}_memories')
    index = load_index_from_storage(storage_context=storage_context)
    index.insert(document)
    index.storage_context.persist(
        persist_dir=f'./memories/{user_name}_memories')
    
    # Specify the path to the text file
    key_summary_path = f'./memories/{user_name}_memories/key_summary.txt'

    # Check if the file already exists
    if os.path.exists(key_summary_path):
        # If the file exists, open it in read ('r') mode
        with open(key_summary_path, 'r') as text_file:
            # Read and print the existing content
            existing_content1 = text_file.read()
    else:
        # If the file doesn't exist, create and write to it in exclusive creation ('x') mode
        with open(key_summary_path, 'x') as text_file:
            text_file.write(summarized_text)

        with open(key_summary_path, 'r') as text_file1:
            # Read and print the existing content
            existing_content1 = text_file1.read()

    def summarize_overall_conversation(key_summary, recent_summary):
        prompt = "The following are the key summary of the past conversations and the summary of the most recent conversation. \
            Please make it into one summary, summarizing the key points. \
            Bear in mind to update information if necessary. \
            For example, if the user talked about an upcoming event in the past and the event has happened in the most recent conversation, \
                the summary should reflect that. \
            If the user mentions about their profile information such as age, gender, occupation and other type of personal information, \
            make sure to include that into the summary. \
            Also, make sure to include friends or family members if mentioned.\
            Make the summary in less than 50 words."
        content = f"key summary: {key_summary.strip()}\nrecent summary: {recent_summary.strip()}"
        return generate(prompt,content)

    #saving overall summary
    if session == 1:
        text_data = existing_content1
    else:
        text_data = summarize_overall_conversation(existing_content1,summarized_text)
    with open(key_summary_path, 'w') as text_file:
        text_file.write(text_data)

    

    