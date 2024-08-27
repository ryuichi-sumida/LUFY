# -*- coding:utf-8 -*-
import os
import math
from llama_index.core import load_index_from_storage, VectorStoreIndex, StorageContext
from llama_index.core.postprocessor import SimilarityPostprocessor
import emoji


# import all the functions
from bmi_utterances import bmi_utterances
from generate import generate
from generate_first_conversation import generate_first_conversation
from relevant_memory_current_context import relevant_memory_current_context
from summarize_wisely import summarize_wisely
from IPE import IPE
from forget_mechanism_wisely import forget_mechanism_wisely

from dotenv import load_dotenv

load_dotenv()

def main():
    # bunch of parameters
    w1,w2,w3,w4,w5 = 0.44776699, -0.2801391,   2.76290708,  1.02800192, -0.01241566
    a1,a2,a3,a4,a5 = 2.1297349437869455,2.8709762924617905,1.5012260479089086,0.62885264522, 2.2459023043571427

    # get the user name
    print("Please Enter Your Name:")
    user_name = input("\nUser Name：")

    # bunch of paths
    path = f"./memories/{user_name}_memories"
    current_session_conversation_path = f"./memories/{user_name}_memories/conversation"
    key_summary_path = f"./memories/{user_name}_memories/key_summary.txt"
    

    # check the current session
    if os.path.exists(current_session_conversation_path):
        session = len(os.listdir(current_session_conversation_path)) + 1
        # get summary from previous session
        with open(key_summary_path, "r") as file:
            key_summary = file.read()
    else:
        session = 1

    if session == 1:
        index = VectorStoreIndex([])
        index.storage_context.persist(persist_dir=path)
        
        print( )

        print("###### Introduction ######")

        print(f"Welcome {user_name}! Now you will talk to LUFY, your AI friend! You can talk about anything you want. If you want to stop, please type 'stop'. Enjoy!" )
        print("first you will be asked to do a short self introduction, an example would be")
        print("Hey there! I'm Ryuichi, a 25 year old, from Japan, currently studying AI. I love to exercise, nice to meet you!")

        print()
        print("###### Here is the actual conversation ######")
        response = f"Hi, I'm LUFY, your AI friend! Very nice to meet you {user_name}! Could you do a short self introduction for me? If you don't mind me asking, could you tell me your age, occupation, nationality?"

    else:
        storage_context = StorageContext.from_defaults(persist_dir=path)
        index = load_index_from_storage(storage_context=storage_context)

        memory_search_query = f"What is the most important thing that has happened to {user_name}? "

        retriever = index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve(memory_search_query)


        # similarity cutoff
        processor = SimilarityPostprocessor(similarity_cutoff=0.6)
        nodes = processor.postprocess_nodes(nodes)

        list_of_node_id_and_score_IPARU_session = []

        number = len(nodes)

        for i in range(number):
            list_of_node_id_and_score_IPARU_session.append((nodes[i].node_id, nodes[i].score, 
            nodes[i].metadata['importance'], nodes[i].metadata['perplexity'], 
            nodes[i].metadata['arousal'], nodes[i].metadata['memory_strength'],
            nodes[i].metadata['unused_count'], nodes[i].metadata['last_recalled_session']))

        list_of_node_id_and_updated_score = []

        for j in list_of_node_id_and_score_IPARU_session:
            importance = j[2] / 27.5
            perplexity = j[3] / 250
            arousal = j[4] / 15
            S = a1 * w1 * importance + a2 * w2 * perplexity + a3 * w3 * arousal + a4 * w4 * (j[5]-1) - a5 * w5 * j[6]
            if S < 0.1:
                S = 0.1
            time_elapsed = session - j[7]
            time_decay = math.exp(-time_elapsed/ (0.5 * S))
            updated_score = j[1] + (time_decay / 10)
            list_of_node_id_and_updated_score.append((j[0], updated_score))


        # sort the list by score
        list_of_node_id_and_updated_score.sort(key=lambda x: x[1], reverse=True)
        list_of_node_id_and_updated_score = list_of_node_id_and_updated_score[:2]
        

        # this is id_, not node_id
        list_retrieved_ids = []

        for a in list_of_node_id_and_updated_score:
            list_retrieved_ids.append(a[0])

        if list_retrieved_ids == []:
            relevant_memory = "None"

        else:
            # get the most relevant memory and update index
            relevant_memory = relevant_memory_current_context(user_name, list_retrieved_ids, session, index)

        
        with open(key_summary_path, "r") as file:
            key_summary = file.read()

        response = generate_first_conversation(relevant_memory, key_summary, user_name,session)

    print()
    print(f"LUFY：{response}")

    while True:
        query = input(f"\n{user_name}：")
        if query.strip():  # Checks if the input is not just whitespace
            break
        else:
            print("Please enter some text.")

    current_conversation = (f'utterance by LUFY: "{response}"' + "," + f' response by {user_name}: "{query}"')

    # saving current conversation
    current_conversation = emoji.demojize(current_conversation)
    document = bmi_utterances(current_conversation, session)
    index.insert(document)
    index.storage_context.persist(
        persist_dir=path
    )
    storage_context = StorageContext.from_defaults(
        persist_dir=path
    )
    index = load_index_from_storage(storage_context=storage_context)
    text_file_path = os.path.join(current_session_conversation_path, f"{session}.txt")
    os.makedirs(current_session_conversation_path, exist_ok=True)
    with open(text_file_path, "a") as text_file:
        text_file.write(current_conversation + "\n")

    # conversation part
    # conversation stops when the user types "stop"
    n = 1
    while True:
        # get the most relevant memory
        n = n + 1
        text = query
        memory_search_query = query
        retriever = index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve(memory_search_query)
        processor = SimilarityPostprocessor(similarity_cutoff=0.75)
        nodes = processor.postprocess_nodes(nodes)
        list_of_node_id_and_score_IPARU_session = []
        number = len(nodes)
        for i in range(number):
            list_of_node_id_and_score_IPARU_session.append((nodes[i].node_id, nodes[i].score, 
            nodes[i].metadata['importance'], nodes[i].metadata['perplexity'], 
            nodes[i].metadata['arousal'], nodes[i].metadata['memory_strength'],
            nodes[i].metadata['unused_count'], nodes[i].metadata['last_recalled_session']))

        list_of_node_id_and_updated_score = []

        for j in list_of_node_id_and_score_IPARU_session:
            importance = j[2] / 27.5
            perplexity = j[3] / 250
            arousal = j[4] / 15
            S = a1 * w1 * importance + a2 * w2 * perplexity + a3 * w3 * arousal + a4 * w4 * (j[5]-1) - a5 * w5 * j[6]
            if S < 0.1:
                S = 0.1
            time_elapsed = session - j[7]
            time_decay = math.exp(-time_elapsed/ (0.5 * S))
            updated_score = j[1] + (time_decay / 10)
            list_of_node_id_and_updated_score.append((j[0], updated_score))
        list_of_node_id_and_updated_score.sort(key=lambda x: x[1], reverse=True)
        list_of_node_id_and_updated_score = list_of_node_id_and_updated_score[:2]
        # this is id_, not node_id
        list_retrieved_ids = []
        for a in list_of_node_id_and_updated_score:
            list_retrieved_ids.append(a[0])

        if list_retrieved_ids == []:
            relevant_memory = "None"

        else:
            # get the most relevant memory and update index
            relevant_memory = relevant_memory_current_context(user_name, list_retrieved_ids, session, index)

        

        if os.path.exists(key_summary_path):
            with open(key_summary_path, "r") as file:
                key_summary = file.read()

        else:
            key_summary = "None"

        #get the recent utterances
        if n > 3:
            text_file_path = os.path.join(current_session_conversation_path, f"{session}.txt")
            with open(text_file_path, "r") as text_file:
                recent_utterances = text_file.read()

            lines = recent_utterances.split('\n')

            # Getting the last 3 lines
            last_3_lines = lines[-3:]
            last_3_lines = ','.join(last_3_lines)
            text = last_3_lines

        response = generate(
            relevant_memory, text, key_summary, user_name
        )

        print()
        print(f"LUFY：{response}")
        n = n + 1
        while True:
            query = input(f"\n{user_name}：")
            if query.strip():  # Checks if the input is not just whitespace
                break
            else:
                print("Please enter some text.")

        if query.strip() == "stop":
            break

        current_conversation = (
        f'utterance by LUFY: "{response}"'
        + ","
        + f' response by {user_name}: "{query}"'
    )
        #getting rid of emojis
        current_conversation = emoji.demojize(current_conversation)

        # save index to disk
        document = bmi_utterances(current_conversation, session)
        index.insert(document)
        index.storage_context.persist(persist_dir=path)
        storage_context = StorageContext.from_defaults(
        persist_dir=path
    )
        index = load_index_from_storage(storage_context=storage_context)  

        os.makedirs(current_session_conversation_path, exist_ok=True)
        # saving conversation to the current session
        text_file_path = os.path.join(current_session_conversation_path, f"{session}.txt")
        with open(text_file_path, "a") as text_file:
            text_file.write(current_conversation + "\n")
            
    IPE(user_name, session)
    summarize_wisely(user_name, session)

    # after session delete some memories
    session1 = session + 1

    forget_mechanism_wisely(user_name, session1,w1,w2,w3,w4,w5)


if __name__ == "__main__":
    main()