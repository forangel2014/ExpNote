import pandas as pd
import json
import numpy as np
from utils import *

task_instruction = """
SYSTEM: Your task is to identify the exactly one entity of type vaccine, disease, drug, or symptom in the given text. Returns in the format of (entity, type).
"""

expnote_train = """
SYSTEM: case 1.
SYSTEM: Identify an entity of type vaccine, disease, drug, or symptom in this text. Returns in the format of (entity, type). Do not include "covid" in the answer.
@celliottability @fordnation @jkwan_md @BrooksFallis I am devastated beyond words . Just found out that my son tested positive for covid . He had a transplant last May . So far he is just nauseous but we all know that with the variants that could change on a dime .
ASSISTANT: ANSWER[(transplant, symptom)]
SYSTEM: Your answer is (transplant, symptom), however, the correct answer is (nauseous, symptom).
ASSISTANT: THINK[I made a mistake, transplant is not the desired entity, and nauseous is a symptom.]
SYSTEM: OK.
ASSISTANT: NOTE[transplant]: "transplant" is not a desired entity.
SYSTEM: OK.
ASSISTANT: NOTE[nauseous]: "nauseous" is a symptom.
SYSTEM: OK.
SYSTEM: case 2.
SYSTEM: Identify an entity of type vaccine, disease, drug, or symptom in this text. Returns in the format of (entity, type). Do not include "covid" in the answer.
@yeahrightgirlhg Except it absolutely is . mRNA vaccines have been around since the early 90 ' s and are well researched . Meanwhile you know the real risks of Covid and to argue that the vaccine is more dangerous than the virus just has no basis in reality .
ASSISTANT: ANSWER[(vaccine, vaccine)]
SYSTEM: Your answer is (vaccine, vaccine), however, the correct answer is (mRNA vaccines, vaccine).
ASSISTANT: THINK[I made a mistake, vaccine is not the desired entity, and mRNA vaccines is a vaccine.]
SYSTEM: OK.
ASSISTANT: NOTE[vaccine]: "vaccine" is not a desired entity.
SYSTEM: OK.
ASSISTANT: NOTE[mRNA vaccines]: "mRNA vaccines" is a vaccine.
SYSTEM: OK.
"""


#SYSTEM: This is testing stage, need to think whether to refer to the given experience and finally answer the question correctly, here are several examples.

expnote_test_experience = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: Identify an entity of type vaccine, disease, drug, or symptom in this text. Returns in the format of (entity, type). Do not include "covid" in the answer.
hot take : coronavirus is the sniffles.
SYSTEM: No relevant experience.
ASSISTANT: THINK[Seems there is no relevant experience about this question, I need to rely on myself.]
SYSTEM: OK.
ASSISTANT: THINK[sniffles seems to be a symptom, so the answer is (sniffles, symptom)]
SYSTEM: OK.
ASSISTANT: ANSWER[(sniffles, symptom)]
SYSTEM: case 2.
SYSTEM: Identify an entity of type vaccine, disease, drug, or symptom in this text. Returns in the format of (entity, type). Do not include "covid" in the answer.
@pattpattpattyo @AaronSuttles @fluffopotamus88 Pneumonia due to COVID just got intubated check up on @RandRCigars for updates as well.
SYSTEM: Here are several possible relevant experience:
SYSTEM: "pneumonia" is a disease.
SYSTEM: "COVID" is not a desired entity.
ASSISTANT: THINK[Based on these experiences, "pneumonia" is a disease, "COVID" is not a desired entity. So the answer is (pneumonia, disease).]
SYSTEM: OK.
ASSISTANT: ANSWER[(pneumonia, disease)]
"""

expnote_test_case = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: Identify an entity of type vaccine, disease, drug, or symptom in this text. Returns in the format of (entity, type). Do not include "covid" in the answer.
hot take : coronavirus is the sniffles.
SYSTEM: No relevant experience.
ASSISTANT: THINK[Seems there is no relevant experience about this question, I need to rely on myself.]
SYSTEM: OK.
ASSISTANT: THINK[sniffles seems to be a symptom, so the answer is (sniffles, symptom)]
SYSTEM: OK.
ASSISTANT: ANSWER[(sniffles, symptom)]
SYSTEM: Your answer is correct.
SYSTEM: case 2.
SYSTEM: Identify an entity of type vaccine, disease, drug, or symptom in this text. Returns in the format of (entity, type). Do not include "covid" in the answer.
@pattpattpattyo @AaronSuttles @fluffopotamus88 Pneumonia due to COVID just got intubated check up on @RandRCigars for updates as well.
SYSTEM: Here are several possible relevant experience:
SYSTEM: Identify an entity of type vaccine, disease, drug, or symptom in this text. Returns in the format of (entity, type). Do not include "covid" in the answer.
@souleymanesado7 @SpokespersonCHN Spit shit out of your mouth ! The earliest virus came from the United States , where unknown pneumonia appeared and electronic cigarettes were blamed for it . 
Your answer is "(unknown pneumonia, disease)", the correct answer is "(pneumonia, disease)".
SYSTEM: Identify an entity of type vaccine, disease, drug, or symptom in this text. Returns in the format of (entity, type). Do not include "covid" in the answer.
@PepeMatter I want numbers on whether pedophilia is a worse epidemic than Covid . Give me THAT data Joe .
Your answer is "(Covid, disease)", the correct answer is "(pedophilia, disease)".
ASSISTANT: THINK[Based on these experiences, "pneumonia" is a disease, "COVID" is not a desired entity. So the answer is (pneumonia, disease).]
SYSTEM: OK.
ASSISTANT: ANSWER[(pneumonia, disease)]
SYSTEM: Your answer is correct.
"""

few_shot = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: Identify an entity of type vaccine, disease, drug, or symptom in this text. Returns in the format of (entity, type).
hot take : coronavirus is the sniffles.
ASSISTANT: (sniffles, symptom)
SYSTEM: case 2.
SYSTEM: Identify an entity of type vaccine, disease, drug, or symptom in this text. Returns in the format of (entity, type).
@neonsprocket U cannot expect much auditing of death during emergency . But even in close circle I know cases of rampant use of steroid on covid patients . Don ' t blame doctors as thr ws no medicine dr wr fighting a war . Regarding disrepute . Let ' s face fact . He made tradition hugely popular . 3 / n.
ASSISTANT: (steroid, drug)
"""

cot = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: Identify an entity of type vaccine, disease, drug, or symptom in this text. Returns in the format of (entity, type).
hot take : coronavirus is the sniffles.
ASSISTANT: sniffles seems to be a symptom, so the answer is (sniffles, symptom).
SYSTEM: case 2.
SYSTEM: Identify an entity of type vaccine, disease, drug, or symptom in this text. Returns in the format of (entity, type).
@neonsprocket U cannot expect much auditing of death during emergency . But even in close circle I know cases of rampant use of steroid on covid patients . Don ' t blame doctors as thr ws no medicine dr wr fighting a war . Regarding disrepute . Let ' s face fact . He made tradition hugely popular . 3 / n.
ASSISTANT: steroid seems to be a drug, so the answer is (steroid, drug)
"""

reflexion = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: Identify an entity of type vaccine, disease, drug, or symptom in this text. Returns in the format of (entity, type).
hot take : coronavirus is the sniffles.
ASSISTANT: coronavirus seems to be a disease, so the answer is (coronavirus, disease).
SYSTEM: Your answer is incorrect.
ASSISTANT: Reflection: I mistakenly thought coronavirus is a disease, I should try again.
SYSTEM: Identify an entity of type vaccine, disease, drug, or symptom in this text. Returns in the format of (entity, type).
hot take : coronavirus is the sniffles.
ASSISTANT: sniffles seems to be a symptom, so the answer is (sniffles, symptom).
SYSTEM: case 2.
SYSTEM: Identify an entity of type vaccine, disease, drug, or symptom in this text. Returns in the format of (entity, type).
@neonsprocket U cannot expect much auditing of death during emergency . But even in close circle I know cases of rampant use of steroid on covid patients . Don ' t blame doctors as thr ws no medicine dr wr fighting a war . Regarding disrepute . Let ' s face fact . He made tradition hugely popular . 3 / n.
ASSISTANT: death seems to be a symptom, so the answer is (death, symptom)
SYSTEM: Your answer is incorrect.
ASSISTANT: Reflection: I mistakenly thought husband's son is stepson, I should try again.
SYSTEM: Identify an entity of type vaccine, disease, drug, or symptom in this text. Returns in the format of (entity, type).
@neonsprocket U cannot expect much auditing of death during emergency . But even in close circle I know cases of rampant use of steroid on covid patients . Don ' t blame doctors as thr ws no medicine dr wr fighting a war . Regarding disrepute . Let ' s face fact . He made tradition hugely popular . 3 / n.
ASSISTANT: steroid seems to be a drug, so the answer is (steroid, drug)
"""

teachme = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: Identify an entity of type vaccine, disease, drug, or symptom in this text. Returns in the format of (entity, type). Do not include "covid" in the answer.
hot take : coronavirus is the sniffles.
SYSTEM: No relevant experience.
ASSISTANT: sniffles seems to be a symptom, so the answer is (sniffles, symptom).
SYSTEM: case 2.
SYSTEM: Identify an entity of type vaccine, disease, drug, or symptom in this text. Returns in the format of (entity, type). Do not include "covid" in the answer.
@pattpattpattyo @AaronSuttles @fluffopotamus88 Pneumonia due to COVID just got intubated check up on @RandRCigars for updates as well.
SYSTEM: Here are several possible relevant experience:
SYSTEM: Pneumonie Synonym pneumonia. pneumonia IsA respiratory disease
SYSTEM: intubated RelatedTo intubate. unintubated DerivedFrom intubated. intubated FormOf intubate. intubated ExternalURL intubated
ASSISTANT: Based on these experiences, "pneumonia" is a disease, "COVID" is not a desired entity. So the answer is (pneumonia, disease).
"""


reflection = [("THINK[", True),
              ("NOTE[", True),
              ("NOTE[", True),
              ("END", False)]

incorrect = "SYSTEM: Your answer is {}, however, the correct answer is {}.\n"
correct = "SYSTEM: Your answer is correct.\n"

prompts = {
    "task_instruction": task_instruction,
    "few_shot": few_shot,
    "cot": cot,
    "reflexion": reflexion,
    "expnote_train": expnote_train,
    "expnote_test_experience": expnote_test_experience,
    "expnote_test_case": expnote_test_case,
    "incorrect": incorrect,
    "correct": correct,
    "reflection": reflection,
    "teachme": teachme
}

def fetch(data, i):
    sample = data[i]
    text = sample['text']
    if not text[-1] == '.':
        text += '.'
    value = sample['value'].strip()
    #query = "Identify an entity of type vaccine, disease, drug, or symptom in this text. Returns in the format of (entity, type)."
    #query = "What type of entity is \"{}\" in this text? Vaccine, Disease, Drug, or Symptom?".format(value)
    target = '({}, {})'.format(sample['value'], sample['type'])
    question = "Identify an entity of type vaccine, disease, drug, or symptom in this text. Returns in the format of (entity, type). Do not include \"covid\" in the answer.\n" + text
    return question, target, text

def extract_answer(sentence):
    #response.split(' ')[-1].strip('.')
    candidates = ["vaccine", "disease", "drug", "symptom"]
    words = sentence.split('(')[-1]
    words = words.split(')')[0]
    answer = '(' + words + ')'
    # words = [word.strip('.').strip(',').strip() for word in words]
    # words_in_candidates = [word for word in words if word in candidates]
    # answer = words_in_candidates[-1] if words_in_candidates else words[-1]
    return answer

def filter_medical_entity(data, deduplicated=False):
    filtered_data = []
    types = ["vaccine-related", "vaccine", "disease", "drug", "symptom"]
    simple_values = ["covid", "cough", "vitamin", "sneez", "fever", "flu", "headache", "sick", "sleeping", "vaccin",
                     "tired", "pain", "cold", "cabbage", "aches", "threw up", "throw up"]
    for i in range(len(data)):
        try:
            sample = data.iloc[i]
            text = sample["preprocessed_text"]
            labels = sample["labels"]
            labels = labels.replace("\'", "\"")
            labels = json.loads(labels)
            
            medical_labels = []
            for label in labels:
                if label["type"].lower() in types:
                    medical_labels.append(label)
            
            if len(medical_labels) == 1:
                label = medical_labels[0]
                type = label["type"]
                value = label["value"]
                flag = True
            elif len(medical_labels) > 1:
                label = medical_labels[0]
                type = label["type"]
                value = label["value"]
                flag = True
                idx0 = text.index(value) + len(value)
                idx1 = text.index(medical_labels[1]["value"])
                for i in range(idx0, idx1):
                    if text[i] in [",", ".", "!", "?"]:
                        text = text[:i]
                        continue
            else:
                flag = False
                
            for simple_value in simple_values:
                if simple_value in value.lower():
                    flag = False
                    break
                
            if flag:
                value_tokens = value.split(" ")
                type = type.lower()
                if type in types and len(value_tokens) <= 2:
                    if type == "vaccine-related":
                        type = "vaccine"
                    filtered_data.append({"text": text, "value": value, "type": type})
        except:
            continue
        
    corrected_data = fixing(filtered_data)
    
    if deduplicated:
        deduplicated_data = []
        existing_entity = []
        for data in corrected_data:
            value = data["value"].lower()
            if value not in existing_entity:
                deduplicated_data.append(data)
                existing_entity.append(value)
        
        return deduplicated_data
    
    else:
        return corrected_data

def fixing(filtered_data):
    
    # ERRORs need fixing in this dataset: 
    # pneumonia -> disease
    # acne -> disease
    
    correction_dict = {"pneumonia": "disease", "acne": "disease"}
    hidden_entities = ["infected", "infection", "depressed", "depression", "tired", "tiredness", "asymptomatic", "headache", "steroids",
                     "feverish", "yellow fever", "muscle aches", "neuropathy", "brain fog", "sore throat", "ivermectin", "cough",
                     "cold sores", "hydroxychloroquine", "corona virus", "rashes", "chest tightness", "astrazeneca"]
    
    entities = [data["value"] for data in filtered_data]
    
    all_entities = list(set(entities).union(set(hidden_entities)))
    
    corrected_data = []
    
    for data in filtered_data:
        
        if data["value"].lower() in correction_dict.keys():
            data["type"] = correction_dict[data["value"].lower()]
            
        if sum([entity in data["text"] for entity in all_entities]) > 1:
            continue
        
        else:
            corrected_data.append(data)
        
    return corrected_data


def load_data():
    
    train_path = "./datasets/mets/train+dev.csv"

    train_data = pd.read_csv(train_path)
    
    train_data = filter_medical_entity(train_data, deduplicated=True)
    train_data = train_data[:300]

    test_path = "./datasets/mets/test.csv"

    test_data = pd.read_csv(test_path)
    
    test_data = filter_medical_entity(test_data, deduplicated=True)
    test_data = test_data[:100]

    n_train = len(train_data)
    all_data = train_data + test_data
    random.shuffle(all_data)
    train_data = all_data[:n_train]
    test_data = all_data[n_train:]

    return train_data, test_data 