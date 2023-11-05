import random
import re
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
import json
import uuid
from typing import List

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
WEAVIATE_KEY = os.getenv("WEAVIATE_API_KEY")
WEVIATE_URL = os.getenv("WEAVIATE_URL")


def get_patient_record(seen_ids: List, model: object) -> dict:
    """Generate a synthetic medical record for a patient using Claude API"""
    if seen_ids and random.randint(1, 10) in [1, 2, 3]:
        patient_id = random.sample(seen_ids, 1)
    else:
        patient_id = str(uuid.uuid4())
        seen_ids.append(patient_id)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ""),
            (
                "human",
                """Generate a synthetic medical record for patient {patient_id}. Include date, age,
                  sex, symptoms, diagnoses, medications, vital signs, lab results, allergies, family 
                  history, and social history. Output as a valid JSON without any additional explanations or 
                  formatting. Any JSON keys should have whitespace replaced with underscores""",
            ),
        ]
    )
    chain = prompt | model
    record = chain.invoke({"patient_id": patient_id})
    content = record.content
    pattern = r"```json([\w\W]+?)```"
    code_blocks = re.findall(pattern, content, re.DOTALL)
    my_json = json.loads(code_blocks[0].strip("\n"))
    return {"json": my_json, "seen_ids": seen_ids}


def get_patient_records(num_records: int = 10) -> List[dict]:
    """Loop through the num_records and create a list of patient records by calling Claude API"""
    num_records = num_records
    records = []
    seen_ids = []
    while len(records) < num_records:
        try:
            record = get_patient_record(seen_ids)
            print(record)
            records.append(record["json"])
            seen_ids = record["seen_ids"]
        except Exception as e:
            print(f"exception {e}")
    return records
