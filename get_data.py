import random
import anthropic
import re
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
import json
import uuid
import weaviate
from sentence_transformers import SentenceTransformer
from typing import List

from create_model import get_model
from create_data import *
from create_weaviate_db import *
from create_outputs import *

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
WEAVIATE_KEY = os.getenv("WEAVIATE_API_KEY")
WEVIATE_URL = os.getenv("WEAVIATE_URL")

if __name__ == "__main__":
    model = get_model(model_name="claude-2")
    for i in range(10):
        records = get_patient_records(num_records=10)
        loop_and_insert(
            records=records,
            endpoint=WEVIATE_URL,
            collection_name="Patient",
        )
    unique_patient_ids = get_unique_patientIDs(endpoint=WEVIATE_URL)
    create_summary_database(
        endpoint=WEVIATE_URL,
        collection_name="SummaryTable",
        patient_ids=unique_patient_ids,
    )
