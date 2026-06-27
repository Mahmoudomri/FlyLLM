from flyllm import FlyLLM
model = FlyLLM.from_pretrained("mistralai/Mistral-7B-v0.1")
print(model.generate("What is AI?", max_new_tokens=50))