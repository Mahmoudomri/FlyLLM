from flyllm import FlyLLM
from flyllm.chat import run_chat

model = FlyLLM.load("~/flyllmmodel/Mistral-7B-v0.1")

run_chat(
    model,
    system="You are a helpful Python tutor. Always show code examples.",
    temperature=0.7,
)
