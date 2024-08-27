import json
import sys
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

from perplexity import calculate_perplexity

sys.path.append("./roberta")
from inference_large import emotion_prediction
sys.path.remove("./roberta")


def IPE(user_name, specified_session):
    index_path = f"./memories/{user_name}_memories"

    def generate(prompt, content):
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}"},
            {"role": "system", "content": "Sure, I will do my best to assist you."},
            {"role": "user", "content": f"{content}"},
        ]

        response = client.chat.completions.create(model="gpt-4o", messages=message)
        response = response.choices[0].message.content
        return response

    key_summary_path = f"{index_path}/key_summary.txt"

    if os.path.exists(key_summary_path):
        with open(key_summary_path, "r") as file:
            key_summary = file.read()

    else:
        key_summary = "None"

    def rate_importance(content):
        prompt = f"On a scale of 1 to 10, where 1 is purely mundane (like brushing teeth or making bed) and \
            10 is extremely important (like a break up or college acceptance), please rate the importance of the following conversation. \
            Rate it based on whether it is useful in later conversations or not. \
            Here is the summary of the past conversations: {key_summary} \
            Here is the converastion: {content} \
                    Make sure to output only the number, I do not need any explanations. Just output the number!"
        return generate(prompt, content)

    # adding metadata to the docstore.json

    filepath1 = f"{index_path}/docstore.json"

    # Load the JSON data from your file
    with open(filepath1, "r") as file:
        data1 = json.load(file)

    # Iterate through the data and update "IPE"
    for key, entry in data1["docstore/data"].items():
        first_mentioned_session = entry["__data__"]["metadata"][
            "first_mentioned_session"
        ]
        if first_mentioned_session == specified_session:
            text = entry["__data__"]["text"]

            # Splitting based on the pattern
            parts = re.split(rf"(response by {user_name}: )", text)

            # Concatenate the first two parts to get the first half of the conversation
            context = parts[0] + parts[1]

            # The last part is the second half of the conversation
            actual_utterance = parts[2].strip('"')

            try:
                perplexity = float(calculate_perplexity(context, actual_utterance))
            except:
                perplexity = 160

            if perplexity > 160:
                perplexity = 160

            try:
                importance = rate_importance(actual_utterance)
                importance = float(importance)
            except:
                print()
                print(f"Error occured: {actual_utterance}")
                print()
                importance = 5

            if importance > 10:
                importance = 10

            emotion = emotion_prediction(actual_utterance).cpu().numpy()
            valence = float(emotion[0])
            arousal = float(emotion[1])

            entry["__data__"]["metadata"]["importance"] = importance
            entry["__data__"]["metadata"]["perplexity"] = perplexity
            entry["__data__"]["metadata"]["valence"] = valence
            entry["__data__"]["metadata"]["arousal"] = arousal

            entry["__data__"]["relationships"]["1"]["metadata"]["importance"] = (
                importance
            )
            entry["__data__"]["relationships"]["1"]["metadata"]["perplexity"] = (
                perplexity
            )
            entry["__data__"]["relationships"]["1"]["metadata"]["valence"] = valence
            entry["__data__"]["relationships"]["1"]["metadata"]["arousal"] = arousal

    with open(filepath1, "w") as file:
        json.dump(data1, file, indent=4)

    with open(filepath1, "r") as file:
        data1 = json.load(file)

    for key, entry in data1["docstore/ref_doc_info"].items():
        first_mentioned_session = entry["metadata"]["first_mentioned_session"]
        if first_mentioned_session == specified_session:
            id = entry["node_ids"][0]
            importance = data1["docstore/data"][id]["__data__"]["metadata"][
                "importance"
            ]
            perplexity = data1["docstore/data"][id]["__data__"]["metadata"][
                "perplexity"
            ]
            valence = data1["docstore/data"][id]["__data__"]["metadata"]["valence"]
            arousal = data1["docstore/data"][id]["__data__"]["metadata"]["arousal"]

            entry["metadata"]["importance"] = importance
            entry["metadata"]["perplexity"] = perplexity
            entry["metadata"]["valence"] = valence
            entry["metadata"]["arousal"] = arousal

    # Save the updated data back to the JSON file
    with open(filepath1, "w") as file:
        json.dump(data1, file, indent=4)
