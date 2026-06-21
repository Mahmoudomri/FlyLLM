from flyllm import FlyLLM

# One command — checks cache, downloads if needed, quantizes if needed, loads
model = FlyLLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# Single prompt
print(model.generate("What is AI?"))

# With system prompt
print(model.generate(
    "Write a Python function to reverse a string",
    system="You are a Python expert. Be concise.",
))

# Stream tokens
for token in model.stream("Explain quantum computing simply"):
    print(token, end="", flush=True)
print()

# Multi-turn conversation
model.set_system("You are a helpful assistant.")
r1 = model.chat_turn("What is the capital of France?")
r2 = model.chat_turn("What is its population?")  # remembers context
print(r1, r2)
