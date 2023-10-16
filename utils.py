import os
import logging
import json
import random
import requests

random_seed = 3
random.seed(random_seed)

def concept_net(model, question):

    prompt = "\nPlease write a few words that you hope will be helpful in answering this question through ConceptNet query, separated by commas:\n"
    response = model(question + prompt)
    entities = response.split(",")
        
    os.environ.pop('HTTPS_PROXY')
    os.environ.pop('HTTP_PROXY')
    
    entities = [entity.lower().strip() for entity in entities]
    
    knowledge = retrieve_useful_knowledge(entities)
    
    os.environ['HTTPS_PROXY'] = 'http://cipzhao:cipzhao@210.75.240.136:10800'
    os.environ['HTTP_PROXY'] = 'http://cipzhao:cipzhao@210.75.240.136:10800'
    
    return knowledge

def retrieve_useful_knowledge(entities):
    
    knowledge_ls = []
    
    for entity in entities:
        url = f'http://api.conceptnet.io/c/en/{entity}?limit=10'
        useful_knowledge = {}

        response = requests.get(url)
        data = response.json()

        for edge in data['edges']:
            rel = edge['rel']['label']
            start = edge['start']['label']
            end = edge['end']['label']

            if rel not in useful_knowledge:
                sentence = f"{start} {rel} {end}"
                useful_knowledge[rel] = sentence

        knowledge_ls.append(useful_knowledge)

    knowledge_ls = [knowledge for knowledge in knowledge_ls if len(knowledge)]
    knowledge = [". ".join(list(knowledge.values())) for knowledge in knowledge_ls]

    return knowledge

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def generate_exp_path(args):
    exp_path = "./exp/{}/{}/{}".format(args.dataset, args.model, args.setting)
    if args.setting == "expnote":
        if args.content != "experience":
            exp_path += f"_{args.content}"
        if args.exp_type != "all":
            exp_path += f"_{args.exp_type}"
        if args.disabled:
            exp_path += "_disabled"
    exp_path += "/"
    return exp_path

def evaluate(scores):
    scores_processed = [score for score in list(scores.values()) if score >= 0]
    num_correct = sum(scores_processed)
    num_process = len(scores_processed)
    accuracy = num_correct / num_process if num_process > 0 else 0
    info = "num correct = {}, num process = {}, accuracy = {}".format(num_correct, num_process, accuracy)
    return info

def save_scores(filename, scores):
    with open(filename, 'w') as f:
        json.dump(scores, f)

def load_scores(filename):
    with open(filename, 'r') as f:
        scores = json.load(f)
    return scores

def majority_vote(ls):
    counts = {}
    for element in ls:
        if element in counts:
            counts[element] += 1
        else:
            counts[element] = 1
    
    max_count = max(counts.values())
    max_elements = [k for k, v in counts.items() if v == max_count]
    
    if len(max_elements) == 1:
        return max_elements[0]
    else:
        return random.choice(max_elements)

def set_logger(exp_path, mode):
    logger_name = mode + '_logger'
    logger = logging.getLogger(logger_name)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(exp_path + f'{mode}.log', mode='w')
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger
    
def filter_exp(exp, exp_type):
    if type(exp) == dict:
        new_dict_items = []
        if exp_type == "all":
            new_dict = exp
        elif exp_type == "positive":
            for key, value in exp.items():
                if value["target"].lower() == value["answer"].lower():
                    new_dict_items.append((key, value))
            new_dict = dict(new_dict_items)
        elif exp_type == "negative":
            for key, value in exp.items():
                if value["target"].lower() != value["answer"].lower():
                    new_dict_items.append((key, value))
            new_dict = dict(new_dict_items)
        else:
            raise Exception("Unsupport experience type")
        return new_dict
    else:
        assert type(exp) == list
        new_ls = []
        if exp_type == "all":
            new_ls = exp
        elif exp_type == "positive":
            for value in exp:
                if value["target"].lower() == value["answer"].lower():
                    new_ls.append(value)
        elif exp_type == "negative":
            for value in exp:
                if value["target"].lower() != value["answer"].lower():
                    new_ls.append(value)
        return new_ls