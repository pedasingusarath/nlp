import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "distilgpt2"

print("Loading DistilGPT2...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
tokenizer.pad_token = tokenizer.eos_token

def generate_answer(question, length="short"):
    question = question.strip()
    if not question:
        raise ValueError("Please enter an academic question.")
    if len(question) > 2000:
        raise ValueError("Question must be 2000 characters or fewer.")

    max_tokens = 50 if length == "short" else 90
    prompt = (
        "Explain the following academic topic in simple terms. "
        "Give a clear, concise educational explanation.\n"
        f"Topic: {question}\nExplanation:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=400).to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=True,
            temperature=0.8, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

if __name__ == "__main__":
    while True:
        q = input("\nEnter an academic question (or 'exit'): ")
        if q.lower() == "exit":
            break
        try:
            print("\nGenerated response:\n", generate_answer(q))
        except ValueError as exc:
            print("Error:", exc)
