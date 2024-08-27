from llama_index.core import Document

def bmi_utterances(text,session):

    #this function returns a document object from text (strings)

    document = Document(
        text=text,
        metadata={
            'filetype': 'conversation',
            'memory_strength':1,
            'importance':0,
            'last_recalled_session':session,
            'first_mentioned_session':session,
            'valence':0,
            'arousal':0,
            'perplexity':0,
            'unused_count':0
        }
    )
    return document