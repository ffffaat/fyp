from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = r"C:\Users\Fatin\OneDrive\Desktop\CAT405\FYP\app\ml_models\dg_final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

prompt = (
    "generate distractors: Tree <sep> "
    "Which visual structure is commonly used to represent hierarchical parent-child relationships? <sep> "
    "A tree represents hierarchical parent-child relationships and is commonly used for structured data."
)

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
outputs = model.generate(**inputs, max_new_tokens=96, num_beams=4)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result)