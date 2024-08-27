import json
import random
import math
from llama_index.core import StorageContext, load_index_from_storage 
import numpy as np
import os
import math

from dotenv import load_dotenv

load_dotenv()

# this is a function to delete memories for a specified session

def forget_mechanism_wisely(user_name, session,w1,w2,w3,w4,w5):
    #to adjust average
    a1,a2,a3,a4,a5 = 2.1297349437869455,2.8709762924617905,1.5012260479089086,0.62885264522, 2.2459023043571427

    # session = after session

    current_session_conversation_path = f"./memories/{user_name}_memories/conversation"
    filepath1 = f"./memories/{user_name}_memories/docstore.json"

    with open(filepath1, "r") as file:
        data1 = json.load(file)

    Number_of_memories = 0
    deleted_memories = 0

    total_probability = 0
    total_S = 0

    list_of_id_and_probability = []

    # Iterate through the data and update "IPE" 
    for key, entry in data1["docstore/data"].items():

        Number_of_memories += 1

        text = entry["__data__"]["text"]
        importance = entry["__data__"]["metadata"]["importance"]    
        perplexity = entry["__data__"]["metadata"]["perplexity"]
        arousal = entry["__data__"]["metadata"]["arousal"]
        memory_strength = entry["__data__"]["metadata"]["memory_strength"]
        unused_count = entry["__data__"]["metadata"]["unused_count"]
        id = entry["__data__"]["relationships"]["1"]["node_id"]
        last_recalled_session = entry["__data__"]["metadata"]["last_recalled_session"]

        time_elapsed = session - last_recalled_session

        importance = importance/ 27.5
        perplexity = perplexity / 400
        arousal = arousal / 15

        S = a1 * w1 * importance + a2 * w2 * perplexity + a3 * w3 * arousal + a4 * w4 * (memory_strength-1) - a5 * w5 * unused_count

        if S < 0.1:
            S = 0.1

        probability = math.exp(-time_elapsed/ (0.33 * S))

        total_probability += probability
        total_S += S

        list_of_id_and_probability.append((id,probability))

    average_probability = total_probability/Number_of_memories

    if average_probability > 0.1 and average_probability < 0.15:
        pass
    elif average_probability < 0.1:
        average_probability = 0.1
    else:
        average_probability = 0.15

    average_probability = average_probability
    
    Number_of_memories_to_keep = math.ceil((average_probability * Number_of_memories) + session - 2)

    list_of_id_and_probability.sort(key=lambda x: x[1], reverse=True)

    list_of_id_and_probability = list_of_id_and_probability[:Number_of_memories_to_keep]

    #make a list of the top 10% of memories to keep
    list_of_id_to_keep = []
    for id, probability in list_of_id_and_probability:
        list_of_id_to_keep.append(id)

    #if id is not in the list of id to keep, delete it
    for key, entry in data1["docstore/data"].items():

        id = entry["__data__"]["relationships"]["1"]["node_id"]

        if id not in list_of_id_to_keep:
            deleted_memories += 1
            storage_context = StorageContext.from_defaults(persist_dir=f"./memories/{user_name}_memories")
            index = load_index_from_storage(storage_context=storage_context)
            index.delete_ref_doc(f"{id}", delete_from_docstore=True)
            index.storage_context.persist(persist_dir=f"./memories/{user_name}_memories")
            index = load_index_from_storage(storage_context=storage_context)

    ### Deletion done, now checking actual numbers ########

    with open(filepath1, "r") as file:
        data1 = json.load(file)

    Number_of_memories_after_deletion = 0

    # Iterate through the data and update "IPE" 
    for key, entry in data1["docstore/data"].items():

        Number_of_memories_after_deletion += 1

    #write these numbers to a .txt file
    filepath2 = f"./memories/{user_name}_memories/LUFY_count.txt"
    with open(filepath2, "a") as file:
        file.write(f"Current session: {session-1}\n")
        file.write(f"Number of memories before deletion: {Number_of_memories}\n")
        file.write(f"Number of memories deleted: {deleted_memories}\n")
        file.write(f"Number of memories after deletion: {Number_of_memories_after_deletion}\n")