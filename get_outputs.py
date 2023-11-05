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
    filtered_pids, summary = filter_for_symptoms(
        patientID="patientID",
        symptoms="symptoms",
        endpoint=WEVIATE_URL,
        num_results=10,
        t1_collection_name="Patient",
        t2_collection_name="SummaryTable",
    )
    final_diagnoses_and_medication = get_final_output(response=filtered_pids, model=model)
    pesonalised_summary = create_natural_language_summary(
        summary=summary,
        final_output=final_diagnoses_and_medication,
        model=model,
        target_comprehension="layman",
    )
