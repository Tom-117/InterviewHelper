import os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import torch
import sys

def load_local_models():
    try:
        # CV információkinyerő model
        roberta_path = "./models/roberta-base-squad2"
        if not os.path.exists(roberta_path):
            print("Downloading RoBERTa model...")
            roberta_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
            roberta_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
            roberta_tokenizer.save_pretrained(roberta_path)
            roberta_model.save_pretrained(roberta_path)
        else:
            roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_path)
            roberta_model = AutoModelForQuestionAnswering.from_pretrained(roberta_path)
# Kérdésgeneráló model
        mistral_path = "./models/mistral-7b-instruct"
        if not os.path.exists(mistral_path):
            print("Downloading Mistral model...")
            mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
            mistral_model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            mistral_tokenizer.save_pretrained(mistral_path)
            mistral_model.save_pretrained(mistral_path)
        else:
            mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_path)
            mistral_model = AutoModelForCausalLM.from_pretrained(
                mistral_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

        
        if torch.cuda.is_available():
            mistral_model = mistral_model.cuda()

        # Értékelő model
        embedder_path = "./models/all-MiniLM-L6-v2"
        if not os.path.exists(embedder_path):
            print("Downloading Sentence Transformer model...")
            embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embedder.save(embedder_path)
        else:
            embedder = SentenceTransformer(embedder_path)

        return roberta_tokenizer, roberta_model, mistral_tokenizer, mistral_model, embedder

    except Exception as e:
        print(f"Error loading models: {str(e)}")
        sys.exit(1)

