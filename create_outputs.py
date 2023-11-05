from dotenv import load_dotenv
import os
from langchain.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
import weaviate
import json

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
WEAVIATE_KEY = os.getenv("WEAVIATE_API_KEY")
WEVIATE_URL = os.getenv("WEAVIATE_URL")


def search_for_similar_summaries(endpoint: str, patientID: str, num_results: int = 10) -> tuple:
    """Given a patients summary, return a list of similar summaries including their diagnoses and medications. Give a confidence
    score for the probability that the patient has the same diagnosis and medications as the similar patient"""
    # Get the patient summary
    client = weaviate.Client(url=endpoint)
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
    patient_summary = filtered_data[0]["summary"]
    patient_vector = filtered_data[0]["vector"]  # Potentially correct up to this point
    # Search for similar summaries
    query = {
        "Aggregate": {
            "Things": [
                {
                    "similarSummaries": {  # Is this mean to be the collection name?
                        "certainty": 0.8,
                        "limit": num_results,
                        "vector": patient_vector,
                    }
                }
            ]
        }
    }
    result = client.query(query)
    # Extract the matching Things
    similar_summaries = result["data"]["Aggregate"]["Things"][0]["similarSummaries"]  # Also need to change this if so
    print("check the format of similar_summaries to see how to call it")
    print(similar_summaries)
    pids = [s["patientID"] for s in similar_summaries["result"]]
    return (pids, patient_summary)


def filter_for_symptoms(
    patientID: str, symptoms: str, endpoint: str, num_results: int, t1_collection_name: str, t2_collection_name: str
) -> tuple:
    """Function that takes patientID, and a list of symptoms and looks up the patient summary based on ID,
    Then calls the search_for_similar_summaries function to return a list of similar summaries and suggested diagnoses/medications
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    symptoms_vector = model.encode(symptoms)
    relevant_summary_pids, summary = search_for_similar_summaries(endpoint, patientID, num_results, t2_collection_name)
    print(relevant_summary_pids)
    client = weaviate.Client(url=endpoint)
    response = (
        client.query.get(t1_collection_name, ["symptoms"])
        .with_where({"path": [patientID], "operator": "In", "valueString": relevant_summary_pids})
        .with_near_vector(
            {
                "vector": symptoms_vector,
            }
        )
        .with_limit(10)
        .with_additional(["distance"])
        .do()
    )
    return (response, summary)


def get_final_output(response: tuple, model: object) -> dict:
    """Function that takes the response from the filter_for_symptoms function and returns a final output
    with the patientID, symptoms, diagnoses, medications, and distance"""
    final_output = []
    for i in response["data"]["Get"]["Things"]:
        patientID = i["patientID"]
        symptoms = i["symptoms"]
        diagnoses = i["diagnoses"]
        medications = i["medications"]
        distance = i["distance"]
        final_output.append(
            {
                "patientID": patientID,
                "symptoms": symptoms,
                "diagnoses": diagnoses,
                "medications": medications,
                "distance": distance,
            }
        )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """"""),
            (
                "human",
                """You will be provided with a list of diagnoses and medications of patients
                 who experienced similar symptoms. Based on how often these occur, provide the most likely 
                 diagnoses and medications. The output should be JSON format, with medications grouped with 
                 the diagnoses they treat. Do not include any explanation or other information in the output 
                 other than the JSON. 
                 <example_output>{"diagnoses": \[{"diagnosis": "treatment"}\]}</example_output>
                <records>"""
                + f"""{final_output}"""
                + """</records>
                Assistant: {"diagnoses":[]}""",
            ),
        ]
    )
    chain = prompt | model
    final_output = chain.invoke().content
    return final_output


def create_natural_language_summary(
    summary: str, final_output: dict, model: object, target_comprehension: str = "non-specialist audiences"
) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"""You are an expert at communicating complex medical topics to {target_comprehension}"""),
            (
                "human",
                f"""Create a summary of an individual's medical history and likely diagnoses and treatments.
                medical history summary: "{summary}"
                likely diagnoses and treatments: {json.dumps(final_output)}.
                If a particular language is specified, translate the summary into that language.""",
            ),
        ]
    )

    chain = prompt | model
    translated_output = chain.invoke().content
    return translated_output
