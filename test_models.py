from src.models.llms import Models

prompt = "Hi! Respond with one sentence."

for model in Models.ALL:
    print(model.invoke(prompt).content)
