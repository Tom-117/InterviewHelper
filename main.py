import os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import torch
import sys

torch.cuda.empty_cache()
torch.backends.cuda.max_split_size_mb = 512

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
                 device_map="auto",
                dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            mistral_tokenizer.save_pretrained(mistral_path)
            mistral_model.save_pretrained(mistral_path)
        else:
            mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_path)
            mistral_model = AutoModelForCausalLM.from_pretrained(
                mistral_path,
                device_map="auto",
                dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True
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

def extract_cv_info(tokenizer, model, cv_text, question):
    inputs = tokenizer(question, cv_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits)
    
    answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end+1])
    return answer.strip()

def analyze_cv(tokenizer, model, cv_text):
    cv_analysis_prompt = f"""You are an AI CV analyzer.
    Extract the following structured information from the given CV text:
    1. Technical skills (e.g., programming languages, tools, frameworks)
    2. Work experience (company names, positions, duration)
    3. Achievements or projects
    4. Soft skills (e.g., teamwork, communication)
    5. Certifications or education

    CV:
    {cv_text}

    Return the result in this JSON format:
    {{
      "skills": [...],
      "experience": [...],
      "projects": [...],
      "soft_skills": [...],
      "education": [...]
    }}"""
    
    inputs = tokenizer(cv_analysis_prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.5,
        top_p=0.9,
        do_sample=False,
        repeat_penalty=1.2
        
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_question(tokenizer, model, cv_data, question_type):
    type_prompts = {
        "technical": f"""<s>[INST] Based on these technical skills: {cv_data['skills']},
                        generate a specific technical interview question about their expertise.
                        Focus on their strongest skill. [/INST]</s>""",
        
        "behavioral": f"""<s>[INST] Given this work experience: {cv_data['experience']}
                         and these soft skills: {cv_data['soft_skills']},
                         generate a behavioral interview question about their teamwork or leadership. [/INST]</s>""",
        
        "motivation": f"""<s>[INST] Considering their background: {cv_data['education']}
                         and achievements: {cv_data['projects']},
                         generate a question about their career goals and motivation. [/INST]</s>"""
    }
    
    prompt = type_prompts.get(question_type, type_prompts["technical"])
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.5,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)