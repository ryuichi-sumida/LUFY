# LUFY: A RAG Chatbot That Forgets Unimportant Conversations

This repository contains the code and dataset for the paper titled  
**[Enhancing Long-term RAG Chatbots with Psychological Models of Memory Importance and Forgetting](https://arxiv.org/pdf/2409.12524)**.

**LUFY** is a Retrieval-Augmented Generation (RAG) chatbot that selectively forgets unimportant conversations — managing to forget over 90% of irrelevant content! This approach enables more efficient long-term interaction with reduced memory bloat and improved relevance.

---

## 📚 Dataset

We are releasing the **largest known conversation dataset** between a human and a system, with each conversation spanning approximately **12,000 tokens** or **253 turns** for 17 unique users.

### Dataset Structure

The dataset is released in two configurations:

### 1. `turns`
Each row corresponds to a single dialogue turn.

**Fields**
- `user_name`: Name of the user (may be anonymized)
- `assistant_name`: Name of the assistant persona
- `conversation_id`: Identifier for a conversation session
- `conversation_date`: Date of the conversation (`YYYY-MM-DD`)
- `turn_id`: Turn identifier (used for evidence linking)
- `role`: One of `user`, `assistant`, 'system'(prompt)
- `content`: Text content of the turn

---

### 2. `qa`
Each row corresponds to a question–answer pair derived from the conversations.

**Fields**
- `user_name`
- `assistant_name`
- `conversation_id`
- `conversation_date`
- `question`: Natural-language question
- `answer`: Ground-truth answer
- `evidence_turn_ids`: List of `turn_id`s that support the answer

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


