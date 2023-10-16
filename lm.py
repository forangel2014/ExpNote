import os
import openai
import numpy as np
import torch
import math
from collections import Counter
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    
class LLM():
    
    def __init__(self, model, device):
        
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.device = device

        if model == 'gpt-j':
            print("---------loading model-------------")
            tokenizer_path = "/data/sunwangtao/.cache/huggingface/hub/models--EleutherAI--gpt-j-6B/snapshots/6e35e2148e92edf096e94d39ac2b98ad59e25975"
            model_path = "/data/sunwangtao/.cache/huggingface/hub/models--EleutherAI--gpt-j-6B/snapshots/b71ae8bc86cac13154e03e92b5855203086b722e"
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.local_lm = AutoModelForCausalLM.from_pretrained(
                model_path,
                revision="float16",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(self.device)
            print("---------finish loading model-------------")
    
        elif model == 'flan-t5':
            print("---------loading model-------------")
            tokenizer_path = "/data/sunwangtao/.cache/huggingface/hub/models--google--flan-t5-small/snapshots/f6b63ff0230b8e19027b922964cab639c1c6da9c"
            model_path = "/data/sunwangtao/.cache/huggingface/hub/models--google--flan-t5-small/snapshots/f6b63ff0230b8e19027b922964cab639c1c6da9c"
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.local_lm = AutoModelForSeq2SeqLM.from_pretrained(
                model_path
            ).to(self.device)
            self.ending_idx = 3
            print("---------finish loading model-------------")
    
        elif model == 'chatglm':
            print("---------loading model-------------")
            tokenizer_path = "/data/sunwangtao/.cache/huggingface/hub/models--THUDM--chatglm-6b/snapshots/2449bdc9d85103734ae987bdc94eafa7fec2145d"
            model_path = "/data/sunwangtao/.cache/huggingface/hub/models--THUDM--chatglm-6b/snapshots/35ca52301fbedee885b0838da5d15b7b47faa37c"
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            self.local_lm = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True
            ).half().to(self.device)
            self.ending_idx = 4
            print("---------finish loading model-------------")
            
        elif model == 'llama2':
            print("---------loading model-------------")
            tokenizer_path = "/data/sunwangtao/llama2-cn/llama-2-7b-chat"
            model_path = "/data/sunwangtao/llama2-cn/llama-2-7b-chat"
            self.local_lm = AutoModelForCausalLM.from_pretrained(model_path,device_map='auto',torch_dtype=torch.float16,load_in_8bit=True)
            self.local_lm = self.local_lm.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,use_fast=False)
            self.tokenizer.pad_token = self.tokenizer.encode('\n')[0]#self.tokenizer.eos_token
            self.ending_idx = [self.tokenizer.eos_token_id, 13]
            self.padding_idx = self.tokenizer.eos_token_id
            print("---------finish loading model-------------")            
        else:
            raise Exception("Model not support")
            
    def __call__(self, prompt, stop=["\n"], temperature=0):
        if self.model == 'gpt3':
            response = self.gpt3(prompt, stop)
        elif self.model == 'chatgpt':
            response = self.chatgpt(prompt, stop, temperature)
        else:
            response = self.local(prompt, stop)
        return post_process(response)
    
    def local(self, prompt, stop=["\n"]):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        response = self.local_lm.generate(inputs, 
                                          max_length=inputs.shape[1]+100, 
                                          early_stopping=True,
                                          eos_token_id=self.ending_idx,
                                          pad_token_id=self.padding_idx)
        response = response[0][inputs.shape[1]:]
        response_text = self.tokenizer.decode(response).strip('\n')
        return response_text
        # generator = pipeline("text-generation", tokenizer=self.tokenizer, model=self.local_lm)
        # generated_text = generator(
        #     prompt,
        #     max_length=50,
        #     early_stopping=True,
        #     stop_token="\n",
        #     num_return_sequences=1
        # )

        # print(generated_text[0]["generated_text"])
    
    def gpt3(self, prompt, stop=["\n"]):
        response = openai.Completion.create(
                        model="text-davinci-003",
                        prompt=prompt,
                        temperature=0,
                        max_tokens=100,
                        top_p=0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        stop=stop
                    )
        return response["choices"][0]["text"]
    
    def chatgpt(self, prompt, stop=["\n"], temperature=0):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", 
                messages=messages,
                temperature=0,
                max_tokens=100,
                top_p=0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=stop
            )
        return response["choices"][0]["message"]["content"]

def post_process(response):
    processed_response = response.rstrip('\n.')
    return processed_response
    
class Discriminator(torch.nn.Module):

    def __init__(self, model, device):
        self.device = device
        self.model_name = "bert-base-uncased"
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.local_lm = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.linear = torch.Linear(self.config.hidden_size, 2)
        
    def forward(self, inputs, labels):
        pass

class WordBasedRetriever(torch.nn.Module):

    def __init__(self, args):
        super(WordBasedRetriever, self).__init__()
        self.tables = {}
        self.args = args

    def encode_new_docs(self, tab_name, docs:dict):
        if tab_name not in self.tables.keys():
            self.tables[tab_name] = {"doc_content": [], "doc_key": []}
        docs = list(docs.items())
        doc_key = [doc[0] for doc in docs]
        doc_content = [doc[1] for doc in docs]
        self.tables[tab_name]["doc_key"].extend(doc_key)
        self.tables[tab_name]["doc_content"].extend(doc_content)
    
    def compute_score(self, query, key):
        query = query.lower().split(" ")
        key_words = key.lower().split(" ")
        query = [kw.replace("\"", "").replace(",", "").replace(".", "").strip() for kw in query]
        key_words = [kw.replace("\"", "").replace(",", "").replace(".", "").strip() for kw in key_words]
        score = sum([kw in query for kw in key_words]) / len(key_words)
        return score

    def retrieve(self, tab_name, query, k=5):
        if tab_name not in self.tables.keys():
            return False
        if len(self.tables[tab_name]["doc_content"]) == 0:
            return False
        doc_key = self.tables[tab_name]["doc_key"]
        doc_content = self.tables[tab_name]["doc_content"]
        doc_scores = np.array([self.compute_score(query, key) for key in doc_key])
        idx = np.argsort(-doc_scores)[:k]
        retrieved_ls = [(doc_key[i], doc_scores[i]) for i in idx]
        return self.filter(retrieved_ls)
    
    def filter(self, retrieved_ls):
        left_keys = []
        for key, score in retrieved_ls:
            if self.args.dataset == "clutrr" or self.args.dataset == "clutrr3":
                if score == 1 and len(key.split(" ")) == 2:
                    left_keys.append(key)
            elif self.args.dataset == "lets":
                if score > 0:
                    left_keys.append(key)
            elif self.args.dataset == "mets":
                if score == 1 and "covid" not in key.lower():
                    left_keys.append(key)
            elif self.args.dataset == "obqa":
                if score == 1:
                    left_keys.append(key)
            elif self.args.dataset == "emoji":
                if score == 1:
                    left_keys.append(key)
            else:
                left_keys.append(key)
        return left_keys
    
class BM25Retriever(torch.nn.Module):

    def __init__(self, k1=1.5, b=0.75):
        super(BM25Retriever, self).__init__()
        self.k1 = k1
        self.b = b
        self.tables = {}

    def encode_new_docs(self, tab_name, docs:dict):
        if tab_name not in self.tables.keys():
            self.tables[tab_name] = {"doc_content": [], "doc_key": [], "doc_stats": {}}
        docs = list(docs.items())
        doc_key = [doc[0] for doc in docs]
        doc_content = [doc[1] for doc in docs]
        self.tables[tab_name]["doc_key"].extend(doc_key)
        self.tables[tab_name]["doc_content"].extend(doc_content)
        doc_stats = {}
        for doc in docs:
            tokens = doc[0].lower().split()
            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                if token not in doc_stats:
                    doc_stats[token] = {"df": 0, "doc_freqs": []}
                doc_stats[token]["df"] += 1
                doc_stats[token]["doc_freqs"].append((doc[0], count))
        self.tables[tab_name]["doc_stats"] = doc_stats

    def compute_idf(self, term, tab_name):
        N = len(self.tables[tab_name]["doc_key"])
        if term in self.tables[tab_name]["doc_stats"].keys():
            df = self.tables[tab_name]["doc_stats"][term]["df"]
        else:
            df = 0
        idf = math.log((N - df + 0.5) / (df + 0.5))
        return idf

    def compute_avgdl(self, tab_name):
        doc_content = self.tables[tab_name]["doc_key"]
        total_words = sum([len(doc.split()) for doc in doc_content])
        avgdl = total_words / len(doc_content)
        return avgdl

    def compute_score(self, query, doc, tab_name):
        score = 0
        k1 = self.k1
        b = self.b
        doc_id = doc[0]
        doc_content = doc[1]
        doc_len = len(doc_id.split())
        query_terms = query.lower().split()
        for term in query_terms:
            tf = 0
            if term in self.tables[tab_name]["doc_stats"].keys():
                for doc_freq in self.tables[tab_name]["doc_stats"][term]["doc_freqs"]:
                    if doc_freq[0] == doc_id:
                        tf = doc_freq[1]
                        break
            idf = self.compute_idf(term, tab_name)
            nominator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / self.compute_avgdl(tab_name)))
            score += idf * (nominator / denominator)
        return score

    def retrieve(self, tab_name, query, k=5):
        if tab_name not in self.tables.keys():
            return False
        if len(self.tables[tab_name]["doc_content"]) == 0:
            return False
        doc_key = self.tables[tab_name]["doc_key"]
        doc_content = self.tables[tab_name]["doc_content"]
        doc_scores = []
        for i in range(len(doc_content)):
            score = self.compute_score(query, (doc_key[i], doc_content[i]), tab_name)
            doc_scores.append((doc_key[i], score))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc[0] for doc in doc_scores[:k]]
    
class DenseRetriever(torch.nn.Module):

    def __init__(self, model_name="bert-base-uncased", device=4):
        super(DenseRetriever, self).__init__()
        self.device = device
        self.model_name = model_name
        self.model = SentenceTransformer('msmarco-distilbert-base-v3').to(self.device)
        self.tables = {}

    def encode_new_docs(self, tab_name, docs:dict):
        if tab_name not in self.tables.keys():
            self.tables[tab_name] = {"doc_content": [], "doc_reps": [], "doc_key": []}
        docs = list(docs.items())
        doc_key = [doc[0] for doc in docs]
        doc_content = [doc[1] for doc in docs]
        doc_reps = [self.encode_sentence(key) for key in doc_key]
        self.tables[tab_name]["doc_key"].extend(doc_key)
        self.tables[tab_name]["doc_content"].extend(doc_content)
        self.tables[tab_name]["doc_reps"].extend(doc_reps)

    def encode_sentence(self, sentence):
        #res = self.model.encode(sentence, device=self.device)
        self.model.eval()
        features = self.model.tokenize([sentence])
        features["input_ids"] = features["input_ids"].to(self.device)
        features["attention_mask"] = features["attention_mask"].to(self.device)
        out_features = self.model(features)["sentence_embedding"]
        return out_features

    def retrieve(self, tab_name, query, k=5):
        if tab_name not in self.tables.keys():
            return False
        if len(self.tables[tab_name]["doc_content"]) == 0:
            return False
        doc_reps = torch.cat(self.tables[tab_name]["doc_reps"], dim=0)
        doc_content = self.tables[tab_name]["doc_content"]
        doc_key = self.tables[tab_name]["doc_key"]
        query_vector = self.encode_sentence(query)
        scores = torch.mm(query_vector, doc_reps.T)[0]
        idx = torch.argsort(scores, descending=True)[:k]
        return [doc_key[i] for i in idx]
    
    def forward(self, inputs, labels):
        pass
