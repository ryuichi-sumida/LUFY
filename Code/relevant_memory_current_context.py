# get the text data from list retrieved ids and make some changes to the metadatas
import json
from llama_index.core import Document


def relevant_memory_current_context(user_name, list_retrieved_ids, session, index):
    if len(list_retrieved_ids) > 0:
        # get the most relevant memory
        filepath1 = (
            f"./memories/{user_name}_memories/docstore.json"
        )

        with open(filepath1, "r") as file:
            data1 = json.load(file)

        text = data1["docstore/data"][list_retrieved_ids[0]]["__data__"]["text"]
        relevant_memory = text

        id = data1["docstore/data"][list_retrieved_ids[0]]["__data__"]["id_"]
        # print(f"here is the id of the most relevant memory: {id}")

        # updating memory strength and last recalled session

        filetype = data1["docstore/data"][list_retrieved_ids[0]]["__data__"][
            "metadata"
        ]["filetype"]
        memory_strength = (
            data1["docstore/data"][list_retrieved_ids[0]]["__data__"]["metadata"][
                "memory_strength"
            ]
            + 1
        )
        importance = data1["docstore/data"][list_retrieved_ids[0]]["__data__"][
            "metadata"
        ]["importance"]
        last_recalled_session = session
        first_mentioned_session = data1["docstore/data"][list_retrieved_ids[0]][
            "__data__"
        ]["metadata"]["first_mentioned_session"]
        valence = data1["docstore/data"][list_retrieved_ids[0]]["__data__"]["metadata"][
            "valence"
        ]
        arousal = data1["docstore/data"][list_retrieved_ids[0]]["__data__"]["metadata"][
            "arousal"
        ]
        perplexity = data1["docstore/data"][list_retrieved_ids[0]]["__data__"][
            "metadata"
        ]["perplexity"]
        unused_count = data1["docstore/data"][list_retrieved_ids[0]]["__data__"][
            "metadata"
        ]["unused_count"]

        updated_document = Document(
            text=text,
            metadata={
                "filetype": filetype,
                "memory_strength": memory_strength,
                "importance": importance,
                "last_recalled_session": last_recalled_session,
                "first_mentioned_session": first_mentioned_session,
                "valence": valence,
                "arousal": arousal,
                "perplexity": perplexity,
                "unused_count": unused_count,
            },
        )

        index.insert(updated_document)
        index.storage_context.persist(
            persist_dir=f"./memories/{user_name}_memories"
        )

        # delete the old one

        id_to_delete = data1["docstore/data"][list_retrieved_ids[0]]["__data__"][
            "relationships"
        ]["1"]["node_id"]
        index.delete_ref_doc(f"{id_to_delete}", delete_from_docstore=True)
        index.storage_context.persist(
            persist_dir=f"./memories/{user_name}_memories"
        )

        if len(list_retrieved_ids) > 1:
            id = data1["docstore/data"][list_retrieved_ids[1]]["__data__"]["id_"]
            # print(f"here is the id of the second most relevant memory: {id}")
            # updating unused count
            text = data1["docstore/data"][list_retrieved_ids[1]]["__data__"]["text"]

            filetype = data1["docstore/data"][list_retrieved_ids[1]]["__data__"][
                "metadata"
            ]["filetype"]
            memory_strength = data1["docstore/data"][list_retrieved_ids[1]]["__data__"][
                "metadata"
            ]["memory_strength"]
            importance = data1["docstore/data"][list_retrieved_ids[1]]["__data__"][
                "metadata"
            ]["importance"]
            last_recalled_session = data1["docstore/data"][list_retrieved_ids[1]][
                "__data__"
            ]["metadata"]["last_recalled_session"]
            first_mentioned_session = data1["docstore/data"][list_retrieved_ids[1]][
                "__data__"
            ]["metadata"]["first_mentioned_session"]
            valence = data1["docstore/data"][list_retrieved_ids[1]]["__data__"][
                "metadata"
            ]["valence"]
            arousal = data1["docstore/data"][list_retrieved_ids[1]]["__data__"][
                "metadata"
            ]["arousal"]
            perplexity = data1["docstore/data"][list_retrieved_ids[1]]["__data__"][
                "metadata"
            ]["perplexity"]
            unused_count = (
                data1["docstore/data"][list_retrieved_ids[1]]["__data__"]["metadata"][
                    "unused_count"
                ]
                + 1
            )

            updated_document1 = Document(
                text=text,
                metadata={
                    "filetype": filetype,
                    "memory_strength": memory_strength,
                    "importance": importance,
                    "last_recalled_session": last_recalled_session,
                    "first_mentioned_session": first_mentioned_session,
                    "valence": valence,
                    "arousal": arousal,
                    "perplexity": perplexity,
                    "unused_count": unused_count,
                },
            )

            index.insert(updated_document1)
            index.storage_context.persist(
                persist_dir=f"./memories/{user_name}_memories"
            )

            # delete the old one

            id_to_delete = data1["docstore/data"][list_retrieved_ids[1]]["__data__"][
                "relationships"
            ]["1"]["node_id"]
            index.delete_ref_doc(f"{id_to_delete}", delete_from_docstore=True)
            index.storage_context.persist(
                persist_dir=f"./memories/{user_name}_memories"
            )

        return relevant_memory
