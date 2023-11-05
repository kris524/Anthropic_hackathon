from dotenv import load_dotenv
import os
from langchain.prompts import ChatPromptTemplate
import json
from typing import List
import weaviate
from sentence_transformers import SentenceTransformer

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
WEAVIATE_KEY = os.getenv("WEAVIATE_API_KEY")
WEVIATE_URL = os.getenv("WEAVIATE_URL")


def loop_and_insert(records: object, endpoint: str, collection_name: str) -> None:
    """Insert list of JSON into a weaviate database and embed only the symptoms"""
    client = weaviate.Client(
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_KEY),
        url=endpoint,  # e.g. "https://some-endpoint.weaviate.network/",  # Replace with your endpoint
    )
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Prepare a batch process
    client.batch.configure(batch_size=10)  # Configure batch
    with client.batch as batch:
        # Batch import all Questions
        for record in records:
            batch.add_data_object(
                record,
                collection_name,
                vector=model.encode(", ".join(record["symptoms"])),  # CHANGE COLLECTION NAME
            )


def filter_individual_patientID(weaviate_url: str, patientID: str) -> dict:
    client = weaviate.Client(weaviate_url)
    filter = {
        "path": ["Things", "patientID"],  # Things appears to be a dynamic name?
        "operator": "Equal",
        "valueString": patientID,
    }
    # Construct a GraphQL query with a filter
    query = {"Get": {"Things": ["*"]}, "Where": filter}
    result = client.query(query)
    # Extract the matching Things
    filtered_data = result["data"]["Get"]["Things"]
    return filtered_data


def get_unique_patientIDs(weaviate_url: str) -> List[str]:
    client = weaviate.Client(weaviate_url)
    query = {"Get": {"Things": ["patientID"]}}
    result = client.query(query)
    # Extract the matching Things
    filtered_data = result["data"]["Get"]["Things"]
    patient_ids = [i["patientID"] for i in filtered_data]
    return patient_ids


def create_summary_database(
    endpoint: str, collection_name: str, patient_ids: List, doctor_type: str = "General Practicioner (GP)"
) -> None:
    for id in set(patient_ids):
        data = filter_individual_patientID(endpoint, id)
        # Join all the text of the records together into a single string
        id_string = ""
        for record in data:
            joined_text = json.dumps(record)
            id_string += joined_text

        # Use Claude to summarise the record string
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are a medical summariser with experience in summarising complex medical histories for patients
                 into a concise summary that be understood by a medical profressional, in this instance a {doctor_type}""",
                ),
                (
                    "human",
                    """The following is a concatentation of one or more medical records for an individual.
                    Summarise the following concatenated medical record into a single string that captures
                    the variance of the individual concatenated records: {record_string}""",
                ),
            ]
        )
        chain = prompt | model
        summary = chain.invoke({"record_string": id_string}).content

        # Vectorize summary
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vector = model.encode(summary)
        # Create a new object with id, summary, and vector
        new_obj = {"patientID": id, "summary": summary, "vector": vector}
        # Add to new database
        client = weaviate.Client(url=endpoint)
        client.batch.add_data_object(new_obj, collection_name)
    client.batch.execute()
