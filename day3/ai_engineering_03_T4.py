import subprocess
subprocess.run(["pip", "install", "--upgrade", "transformers"])
subprocess.run(["pip", "install", "google-colab-selenium"])
subprocess.run(["pip", "install", "bitsandbytes"])
subprocess.run(["pip", "install", "huggingface_hub"])

# 演習用のコンテンツを取得
subprocess.run(["git", "clone", "https://github.com/matsuolab/lecture-ai-engineering.git"])

# HuggingFace Login
from huggingface_hub import notebook_login

notebook_login()

# CUDAが利用可能ならGPUを、それ以外ならCPUをデバイスとして設定
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random
random.seed(0)

# モデル(Gemma2)の読み込み

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "google/gemma-2-2b-jpn-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )

messages = [
    {"role": "user", "content": "LLMにおけるInference Time Scalingとは？"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=False,
    # temperature=0.6, # If do_sample=True
    # top_p=0.9,  # If do_sample=True
)

response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))

from sentence_transformers import SentenceTransformer

emb_model = SentenceTransformer("infly/inf-retriever-v1-1.5b", trust_remote_code=True)
# In case you want to reduce the maximum length:
emb_model.max_seq_length = 4096

with open("/content/lecture-ai-engineering/day3/data/LLM2024_day4_raw.txt", "r") as f:
  raw_writedown = f.read()

# ドキュメントを用意する。
documents = [text.strip() for text in raw_writedown.split("。")]
print("ドキュメントサイズ: ", len(documents))
print("ドキュメントの例: \n", documents[250])

# Retrievalの実行
question = "LLMにおけるInference Time Scalingとは？"

query_embeddings = emb_model.encode([question], prompt_name="query")
document_embeddings = emb_model.encode(documents)

# 各ドキュメントの類似度スコア
scores = (query_embeddings @ document_embeddings.T) * 100
print(scores.tolist())

topk = 5
for i, index in enumerate(scores.argsort()[0][::-1][:topk]):
  print(f"取得したドキュメント{i+1}: (Score: {scores[0][index]})")
  print(documents[index], "\n\n")

 #回答に役立つ該当の発言はreference[1871]〜に含まれてます。
references = "\n".join(["* " + documents[i] for i in scores.argsort()[0][::-1][:topk]])
messages = [
    {"role": "user", "content": f"[参考資料]\n{references}\n\n[質問] LLMにおけるInference Time Scalingとは？"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=False,
    # temperature=0.6, # If do_sample=True
    # top_p=0.9,  # If do_sample=True
)

response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))

with open("/content/lecture-ai-engineering/day3/data/LLM2024_day4.txt", "r") as f:
  raw_writedown = f.read()

# ドキュメントを用意する。
documents = [text.strip() for text in raw_writedown.split("。")]
print("ドキュメントサイズ: ", len(documents))
print("ドキュメントの例: \n", documents[310])

# Retrievalの実行
question = "LLMにおけるInference Time Scalingとは？"

query_embeddings = emb_model.encode([question], prompt_name="query")
document_embeddings = emb_model.encode(documents)

# 各ドキュメントの類似度スコア
scores = (query_embeddings @ document_embeddings.T) * 100
print(scores.tolist())

topk = 5
for i, index in enumerate(scores.argsort()[0][::-1][:topk]):
  print(f"取得したドキュメント{i+1}: (Score: {scores[0][index]})")
  print(documents[index], "\n\n")

 #回答に役立つ該当の発言はreference[1871]〜に含まれてます。
references = "\n".join(["* " + documents[i] for i in scores.argsort()[0][::-1][:topk]])
messages = [
    {"role": "user", "content": f"[参考資料]\n{references}\n\n[質問] LLMにおけるInference Time Scalingとは？"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=False,
    # temperature=0.6, # If do_sample=True
    # top_p=0.9,  # If do_sample=True
)

response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))

# 前後それぞれ2つずつの文章を一つのドキュメントに追加する。（要は5つの文章集合になる)
references = "\n".join(["* " + "。".join(documents[max(0, i-2): min(i+2, len(documents))]).strip() for i in scores.argsort()[0][::-1][:topk]])
messages = [
    {"role": "user", "content": f"[参考資料]\n{references}\n\n[質問] LLMにおけるInference Time Scalingとは？"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=False,
    # temperature=0.6, # If do_sample=True
    # top_p=0.9,  # If do_sample=True
)

response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))

 #回答に役立つ該当の発言はreference[1871]〜に含まれてます。
references = []
for ref in ["。".join(documents[max(0, i-2): min(i+2, len(documents))]).strip() for i in scores.argsort()[0][::-1][:topk]]:
  messages = [
      {"role": "user", "content": f"与えられた[参考資料]が[質問]に直接関連しているかを、'yes''no'で答えること。[参考資料]\n{ref}\n\n[質問] LLMにおけるInference Time Scalingとは？"},
  ]
  input_ids = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      return_tensors="pt"
  ).to(model.device)

  terminators = [
      tokenizer.eos_token_id,
      tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = model.generate(
      input_ids,
      # max_new_tokens=128,
      eos_token_id=terminators,
      do_sample=False,
      # temperature=0.6, # If do_sample=True
      # top_p=0.9,  # If do_sample=True
  )

  response = outputs[0][input_ids.shape[-1]:]
  response = tokenizer.decode(response, skip_special_tokens=True)
  print("\n\n対象となるドキュメント:\n", ref.replace("。", "。\n"))
  print("\n関連しているかどうか: ", response)

  if "yes" in response.lower():
    references.append(ref)

print(len(references))

 #回答に役立つ該当の発言はreference[1871]〜に含まれてます。
messages = [
    {"role": "user", "content": f"与えられる資料を参考にして回答すること。[参考資料]\n{references}\n\n[質問] LLMにおけるInference Time Scalingとは？"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=False,
    # temperature=0.6, # If do_sample=True
    # top_p=0.9,  # If do_sample=True
)

response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))

# 本来は段落をそのままdocumentsに入れずに一定のサイズに分割した方が良いでしょうが、簡単のために段落をそのまま入れてしまいます。
documents = [text.replace("\n", " ").strip() for text in raw_writedown.split("\n\n")]
print("ドキュメントサイズ: ", len(documents))
print("ドキュメントの例: \n", documents[30])

question = "LLMにおけるInference Time Scalingとは？"

query_embeddings = emb_model.encode([question], prompt_name="query")
document_embeddings = emb_model.encode(documents)

scores = (query_embeddings @ document_embeddings.T) * 100
print(scores.tolist())

# 簡単のためにtop2でやります。結果を見てもらえれば問題なく関連する項目のみ取得できているのが分かるかと思います。
topk = 2
for i, index in enumerate(scores.argsort()[0][::-1][:topk]):
  print(f"取得したドキュメント{i+1}: (Score: {scores[0][index]})")
  print(documents[index], "\n\n")

reference = "\n".join(["* " + documents[i] for i in scores.argsort()[0][::-1][:topk]])

messages = [
    {"role": "user", "content": f"[参考資料]\n{reference}\n\n[質問] LLMにおけるInference Time Scalingとは？"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    # max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=False,
    # temperature=0.6, # If do_sample=True
    # top_p=0.9,  # If do_sample=True
)

response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
