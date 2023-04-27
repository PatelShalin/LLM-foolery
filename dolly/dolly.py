# import torch
# from transformers import pipeline
# import time
# start_time = time.time()
# generate_text = pipeline(model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

# print(generate_text("What are some good apartments in New York City?"))
# print("--- %s seconds ---" % (time.time() - start_time))


from dolly.instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b", device_map="auto", offload_folder="offload")

# # TRAIN
# sequences = [{"instruction": "Which is a species of fish? Tope or Rope", "context": "", "response": "Tope", "category": "classification"}]
# batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

PROMPT = "What is the average rent for a studio apartment in New York City?"

generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
print(generate_text(PROMPT))
print("--- %s seconds ---" % (time.time() - start_time))