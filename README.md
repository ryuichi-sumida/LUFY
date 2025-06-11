# LUFY: A RAG Chatbot That Forgets Unimportant Conversations

This repository contains the code and dataset for the paper titled  
**[Should RAG Chatbots Forget Unimportant Conversations? Exploring Forgetting with Psychological Insights](https://arxiv.org/pdf/2409.12524)**.

**LUFY** is a Retrieval-Augmented Generation (RAG) chatbot that selectively forgets unimportant conversations — managing to forget over 90% of irrelevant content! This approach enables more efficient long-term interaction with reduced memory bloat and improved relevance.

---

## 📚 Dataset

We are releasing the **largest known conversation dataset** between a human and a system, with each conversation spanning approximately **12,000 tokens** or **253 turns** for 17 different users.

Each conversation entry includes:
- Full dialogue history
- Question-Answer (QA) pairs
- Evidence utterances supporting each answer

---

## 💬 Talk to LUFY

### 🛠️ Set Up the Environment

Step1: conda create --name your_env_name python=3.11

Step2: pip install -r requirements.txt

Step3: #Edit the .env file and write your OPENAI API KEY

## Actual conversation step
Step4: Navigate to LUFY/code directory and type "python chat.py" to start the conversation.

Type "stop" to stop the conversation.
Enjoy!

## Additional Information

For more details on the project, including the methodology and results, please refer to our paper [here](https://arxiv.org/pdf/2409.12524) or contact the authors (sumida@sap.ist.i.kyoto-u.ac.jp).


