from langchain.chat_models import ChatAnthropic
from dotenv import load_dotenv
import os

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def get_model(model_name: str) -> ChatAnthropic:
    model = ChatAnthropic(
        model_name=model_name,
        anthropic_api_key=ANTHROPIC_API_KEY,
        max_tokens_to_sample=2000,
    )
    return model
