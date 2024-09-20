# LUFY: A RAG Chatbot that forgets unimportant conversations

This repository contains the code and dataset for the paper titled [Should RAG Chatbots Forget Unimportant Conversations?
Exploring Forgetting with Psychological Insights](https://arxiv.org/pdf/2409.12524). 

LUFY is a RAG-based chatbot designed to forget unimportant conversations, managing to forget over 90% of irrelevant content!

# Talk to LUFY

## Create the enironment necessary to run the files

Step1: conda create --name your_env_name python=3.11

Step2: pip install -r requirements.txt

Step3: #Edit the .env file and write your OPENAI API KEY

## Actual conversation step
Step4: Navigate to LUFY/code directory and type "python chat.py" to start the conversation.

Type "stop" to stop the conversation.
Enjoy!

# Dataset

We are also releasing a new dataset that is 4.5x larger than any existing text-based conversation dataset, structured as follows:
/Dataset/{System Name}/{User Name}/{Session Number}/{Annotation Number}.xlsx

- **System Name**: The chatbot's name, either LUFY, MemoryBank or Naive RAG.
- **User Name**: The name of the user.
- **Session Number**: Either 1, 2, 3 or 4.
- **Annotation Number**: Either 1, 2 or 3.

## Additional Information

For more details on the project, including the methodology and results, please refer to our paper [here](https://arxiv.org/pdf/2409.12524) or contact the authors (sumida@sap.ist.i.kyoto-u.ac.jp).


