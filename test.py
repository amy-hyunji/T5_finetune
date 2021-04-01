import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("deep-learning-analytics/triviaqa-t5-base")

model = AutoModelWithLMHead.from_pretrained("deep-learning-analytics/triviaqa-t5-base")


text = "Mount Everest is found in which mountain range?"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
preprocess_text = text.strip().replace("\n","")
tokenized_text = tokenizer.encode(preprocess_text, return_tensors="pt").to(device)
model = model.to(device)

outs = model.model.generate(
            tokenized_text,
            max_length=10,
            num_beams=2,
            early_stopping=True
           )

dec = [tokenizer.decode(ids) for ids in outs]
print("Predicted Answer: ", dec)