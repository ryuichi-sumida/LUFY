# LUFY: A RAG Chatbot that forgets unimportant conversations

This repository contains the code and dataset for the paper titled [Should RAG Chatbots Forget Unimportant Conversations?
Exploring Forgetting with Psychological Insights](https://baseball.yahoo.co.jp/npb/). 

LUFY is a RAG-based chatbot designed to forget unimportant conversations, managing to forget over 90% of irrelevant content!

## Getting Started

To quickly start using the chatbot, please refer to the [README in the Code directory](./Code/README.md) for detailed instructions on how to set up, run, and interact with LUFY.

## Dataset

We are also releasing a new dataset that is 4.5x larger than any existing text-based conversation dataset, structured as follows:
/Dataset/{System Name}/{User Name}/{Session Number}/{Annotation Number}.xlsx

- **System Name**: The chatbot's name, either LUFY, MemoryBank or Naive RAG.
- **User Name**: The name of the user.
- **Session Number**: Either 1, 2, 3 or 4.
- **Annotation Number**: Either 1, 2 or 3.

## Additional Information

For more details on the project, including the methodology and results, please refer to our paper [here](https://baseball.yahoo.co.jp/npb/).


