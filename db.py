import weaviate
import os
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

WEAVIATE_KEY = os.getenv("WEAVIATE_KEY")
auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_KEY)
client = weaviate.Client(url="https://diagnosis-ai-generator-kk6m9y1x.weaviate.network", auth_client_secret=auth_config)


class_obj = {
    "classes": [
        {
            "class": "SummaryTable",
            "properties": [
                {
                    "name": "patientID",
                    "dataType": ["uuid"],
                }
            ],
        },
        {
            "class": "Patient",
            "description": "Patient medical record",
            "vectorizer": "text2vec-openai",
            "properties": [
                {
                    "name": "patientID",
                    "dataType": ["text"],
                    "tokenization": "word",
                },
                {
                    "name": "age",
                    "dataType": ["int"],
                },
                {
                    "name": "sex",
                    "dataType": ["text"],
                    "tokenization": "word",
                },
                {
                    "name": "symptoms",
                    "dataType": ["text[]"],
                    "tokenization": "word",
                },
                {
                    "name": "diagnoses",
                    "dataType": ["text[]"],
                    "tokenization": "word",
                },
                {
                    "name": "medications",
                    "dataType": ["text[]"],
                    "tokenization": "word",
                },
                {
                    "name": "vitalSigns",
                    "dataType": ["object"],
                    "nestedProperties": [
                        {"dataType": ["text"], "name": "BloodPressure"},
                        {"dataType": ["text"], "name": "HeartRate"},
                        {"dataType": ["text"], "name": "Temperature"},
                    ],
                },
                {
                    "name": "labResults",
                    "dataType": ["text"],
                    "tokenization": "word",
                },
                {
                    "name": "allergies",
                    "dataType": ["text[]"],
                    "tokenization": "word",
                },
                {
                    "name": "familyHistory",
                    "dataType": ["text[]"],
                    "tokenization": "word",
                },
                {
                    "name": "socialHistory",
                    "dataType": ["object"],
                    "nestedProperties": [
                        {"dataType": ["text"], "name": "SmokingStatus"},
                        {"dataType": ["text"], "name": "AlcoholUse"},
                    ],
                },
            ],
        },
    ]
}

pprint(client.schema.get())
# questions = client.collections.get("Patient")
# client.schema.delete_class(cls["class"])
# client.schema.delete_class("Patient")
for cls in class_obj["classes"]:
    # # #     # print(cls)
    client.schema.create_class(cls)
