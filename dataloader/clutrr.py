import pandas as pd
import logging
import argparse
from utils import *

task_instruction = """
SYSTEM: Your task is to answer the question given a short story.
"""

expnote_train = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: [Amy] is having a meal with her father [Jackson]. [Lucy] is happy watching TV with her husband [Jackson]. Who is Amy to Lucy?
ASSISTANT: THINK[Amy is Jackson's daughter, Jackson is Lucy's husband. Amy is Lucy's husband's daughter. so Amy is Lucy's daughter-in-law.]
SYSTEM: OK.
ASSISTANT: ANSWER[Amy is Lucy's daughter-in-law.]
SYSTEM: Your answer is daughter-in-law, however, the correct answer is daughter.
ASSISTANT: THINK[In this case, Amy has a father Jackson, Jackson is the husband of Lucy. The conclusion is that Amy is the daughter of Lucy.]
SYSTEM: OK.
ASSISTANT: THINK[Replace Amy with A, Jackson with B, and Lucy with C. The abstract rule in this case is that if A has a father B, B is the husband of C, then A is the daughter of C.]
SYSTEM: OK.
ASSISTANT: NOTE[father, husband]: if A has a father B, B is the husband of C, then A is the daughter of C.
SYSTEM: case 2.
SYSTEM: [Chris] and his son [Jack] are at a bar waiting for their drinks. [Maria] showed up and sat with her husband [Chris]. Who is Jack to Chris? 
ASSISTANT: THINK[Jack is Chris's son, Chris is Maria's husband. Jack is Maria's husband's son. so Jack is Maria's stepson.]
SYSTEM: OK.
ASSISTANT: ANSWER[Jack is Maria's stepson.]
SYSTEM: Your answer is stepson, however, the correct answer is son.
ASSISTANT: THINK[In this case, Jack is the son of Chris, Chris is the husband of Maria. The conclusion is that Jack is the son of Maria.]
SYSTEM: OK.
ASSISTANT: THINK[Replace Jack with A, Chris with B, and Maria with C. The abstract rule in this case is that if A is the son of B, B is the husband of C, then A is the son of C.]
SYSTEM: OK.
ASSISTANT: NOTE[son, husband]: if A is the son of B, B is the husband of C, then A is the son of C.
SYSTEM: case 3.
SYSTEM: [George] drove her daughter [Amy] to the hospital. [Amy]'s son [Bob] showed up shortly after. Who is Bob to George? 
ASSISTANT: THINK[Amy is George's daughter, Bob is Amy's son. Bob is George's daughter's son. so Bob is George's grandson.]
SYSTEM: OK.
ASSISTANT: ANSWER[Bob is George's grandson.]
SYSTEM: Your answer is correct.
ASSISTANT: THINK[In this case, George has a daughter Amy, Amy has a son Bob. The conclusion is that Bob is the grandson of George.]
SYSTEM: OK.
ASSISTANT: THINK[Replace George with A, Amy with B, and Bob with C. The abstract rule in this case is that if A has a daughter B, B has a son C, then C is the grandson of A.]
SYSTEM: OK.
ASSISTANT: NOTE[daughter, son]: if A has a daughter B, B has a son C, then C is the grandson of A.
"""

expnote_test_experience = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: [Amy]'s brother [Mike] and her went to get ice cream. [Mike] is the proud father of the lovely [Jackson]. Who is Amy to Jackson? 
SYSTEM: No relevant experience.
ASSISTANT: THINK[Seems there is no relevant experience about this question, I need to rely on myself]
SYSTEM: OK.
ASSISTANT: THINK[Mike is Amy's brother, Amy is female, Mike is Jackson's father. Dorophy is Jackson's father's sister. so Amy is Jackson's aunt.]
SYSTEM: OK.
ASSISTANT: ANSWER[Amy is Jackson's aunt.]
SYSTEM: Your answer is correct.
SYSTEM: case 2.
SYSTEM: [Chris] and his son [Jack] are at a bar waiting for their drinks. [Maria] showed up and sat with her husband [Chris]. Who is Jack to Maria? 
SYSTEM: Here are several possible relevant experience:
SYSTEM: if A has a son B, A has a wife C, then B is the son of C.
ASSISTANT: THINK[This experience seems helpful, I should refer to it.]
SYSTEM: OK.
ASSISTANT: THINK[Jack is Chris's son, Chris is Maria's husband. According to this experience, Jack is Maria's son.]
SYSTEM: OK.
ASSISTANT: ANSWER[Jack is Maria's son.]
SYSTEM: Your answer is correct.
SYSTEM: case 3.
SYSTEM: [George] drove her daughter [Amy] to the hospital. [Amy]'s son [Bob] showed up shortly after. Who is Bob to George? 
SYSTEM: Here are several possible relevant experience:
SYSTEM: if A has a daughter B, A has a son C, then B is the brother of C.
ASSISTANT: THINK[This experience doesn't seem to be helpful in answering this question. I need to rely on myself.]
SYSTEM: OK.
ASSISTANT: THINK[Amy is George's daughter, Bob is Amy's son. Bob is George's daughter's son. so Bob is George's grandson.]
SYSTEM: OK.
ASSISTANT: ANSWER[Bob is George's grandson.]
SYSTEM: Your answer is correct.
"""

expnote_test_case = """
SYSTEM: This is testing stage, you will be given several possible relevant experiences along with the question.
You need decide whether these experiences are helpful to this question and whether to refer to them. Here are several examples.
SYSTEM: case 1.
SYSTEM: [Amy]'s brother [Mike] and her went to get ice cream. [Mike] is the proud father of the lovely [Jackson]. Who is Amy to Jackson? 
SYSTEM: No relevant experience.
ASSISTANT: THINK[Seems there is no relevant experience about this question, I need to rely on myself]
SYSTEM: OK.
ASSISTANT: THINK[Mike is Amy's brother, Amy is female, Mike is Jackson's father. Dorophy is Jackson's father's sister. so Amy is Jackson's aunt.]
SYSTEM: OK.
ASSISTANT: ANSWER[Amy is Jackson's aunt.]
SYSTEM: Your answer is correct.
SYSTEM: case 2.
SYSTEM: [Chris] and his son [Jack] are at a bar waiting for their drinks. [Maria] showed up and sat with her husband [Chris]. Who is Jack to Maria? 
SYSTEM: Here are several possible relevant experience:
SYSTEM: [Dorothy] went to dinner with her husband [James]. [James] is so proud of his son. [Dan] is the son\'s name. Who is Dan to Dorothy?
Your answer is "stepson", the correct answer is "son".
ASSISTANT: THINK[This experience seems helpful, I should refer to it.]
SYSTEM: OK.
ASSISTANT: THINK[Jack is Chris's son, Chris is Maria's husband. Based on this experience, Jack is Maria's son.]
SYSTEM: OK.
ASSISTANT: ANSWER[Jack is Maria's son.]
SYSTEM: Your answer is correct.
SYSTEM: case 3.
SYSTEM: [George] drove her daughter [Amy] to the hospital. [Amy]'s son [Bob] showed up shortly after. Who is Bob to George? 
SYSTEM: Here are several possible relevant experience:
SYSTEM: While [James] was at the playground with his young daughter [Theresa], he introduced me to his mom, [Gabrielle]. Who is Theresa to Gabrielle?
Your answer is "granddaughter", the correct answer is "granddaughter".
SYSTEM: [Dwight] has always been strict with his daughter, [Donna]. He feels it will help her become a disciplined adult. [Jason] rushed to see his grandmother, [Donna], in the hospital. Who is Jason to Dwight?
Your answer is "grandson", the correct answer is "father".
SYSTEM: OK.
ASSISTANT: THINK[Amy is George's daughter, Bob is Amy's son. Bob is George's daughter's son. so Bob is George's grandson.]
SYSTEM: OK.
ASSISTANT: ANSWER[Bob is George's grandson.]
SYSTEM: Your answer is correct.
"""

few_shot = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: [Amy]'s brother [Mike] and her went to get ice cream. [Mike] is the proud father of the lovely [Jackson]. Who is Amy to Jackson? 
ASSISTANT: aunt.
SYSTEM: case 2.
SYSTEM: [Chris] and his son [Jack] are at a bar waiting for their drinks. [Maria] showed up and sat with her husband [Chris]. Who is Jack to Chris? 
ASSISTANT: son.
SYSTEM: case 3.
SYSTEM: [Mike] and his daughter [Jennifer] like to read poems together. [Jason] is the proud father of the lovely [Mike]. Who is Jason to Jennifer? 
ASSISTANT: grandfather.
"""

cot = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: [Amy]'s brother [Mike] and her went to get ice cream. [Mike] is the proud father of the lovely [Jackson]. Who is Amy to Jackson? 
ASSISTANT: Mike is Amy's brother, Amy is female, Mike is Jackson's father. Dorophy is Jackson's father's sister. so Amy is Jackson's aunt.
SYSTEM: case 2.
SYSTEM: [Chris] and his son [Jack] are at a bar waiting for their drinks. [Maria] showed up and sat with her husband [Chris]. Who is Jack to Chris? 
ASSISTANT: Jack is Chris's son, Chris is Maria's husband. Jack is Maria's husband's son. so Jack is Maria's son.
SYSTEM: case 3.
SYSTEM: [Mike] and his daughter [Jennifer] like to read poems together. [Jason] is the proud father of the lovely [Mike]. Who is Jason to Jennifer? 
ASSISTANT: Jennifer is Mike's daughter, Jason is Mike's father. Jason is Jennifer's father's father. so Jason is Jennifer's grandfather.
"""

reflexion = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: [Amy]'s brother [Mike] and her went to get ice cream. [Mike] is the proud father of the lovely [Jackson]. Who is Amy to Jackson? 
ASSISTANT: Mike is Amy's brother, Amy is female, Mike is Jackson's father. Dorophy is Jackson's father's sister. so Amy is Jackson's mother.
SYSTEM: Your answer is incorrect.
ASSISTANT: Reflection: I mistakenly thought father's sister is mother, I should try again.
SYSTEM: [Amy]'s brother [Mike] and her went to get ice cream. [Mike] is the proud father of the lovely [Jackson]. Who is Amy to Jackson? 
ASSISTANT: Mike is Amy's brother, Amy is female, Mike is Jackson's father. Dorophy is Jackson's father's sister. so Amy is Jackson's aunt.
SYSTEM: case 2.
SYSTEM: [Chris] and his son [Jack] are at a bar waiting for their drinks. [Maria] showed up and sat with her husband [Chris]. Who is Jack to Chris? 
ASSISTANT: Jack is Chris's son, Chris is Maria's husband. Jack is Maria's husband's son. so Jack is Maria's stepson.
SYSTEM: Your answer is incorrect.
ASSISTANT: Reflection: I mistakenly thought husband's son is stepson, I should try again.
SYSTEM: [Chris] and his son [Jack] are at a bar waiting for their drinks. [Maria] showed up and sat with her husband [Chris]. Who is Jack to Chris? 
ASSISTANT: Jack is Chris's son, Chris is Maria's husband. Jack is Maria's husband's son. so Jack is Maria's son.
"""

teachme = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: [Amy]'s brother [Mike] and her went to get ice cream. [Mike] is the proud father of the lovely [Jackson]. Who is Amy to Jackson? 
SYSTEM: No relevant experience.
ASSISTANT: Mike is Amy's brother, Amy is female, Mike is Jackson's father. Dorophy is Jackson's father's sister. so Amy is Jackson's aunt. Amy is Jackson's aunt.
SYSTEM: case 2.
SYSTEM: [Chris] and his son [Jack] are at a bar waiting for their drinks. [Maria] showed up and sat with her husband [Chris]. Who is Jack to Chris? 
SYSTEM: Here are several possible relevant experience:
SYSTEM: chris RelatedTo englisch. chris HasProperty nuts. Chris IsA a violinist. Chris CapableOf drive to freedom. chris Desires sex. chris FormOf christopher. chrissy DerivedFrom chris
SYSTEM: A jack IsA a lifting device. card RelatedTo jack. diddly-shit Synonym jack. Jack CapableOf deal drugs
SYSTEM: Calophyllum longifolium Synonym Maria
ASSISTANT: Jack is Chris's son, Chris is Maria's husband. According to this experience, Jack is Maria's son. Jack is Maria's son.
"""

reflection = [("THINK[", True),
              ("THINK[", True),
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
    sample = data.iloc[i]
    story = sample['story']
    if not story[-1] == '.':
        story += '.'
    query = sample['query']
    query = query.strip(r"[\", (, ), \']").split("', '")[::-1]
    query = "Who is {} to {}?".format(*query)
    target = sample['target']
    question = story + ' ' + query
    return question, target, story

def extract_answer(sentence):
    #response.split(' ')[-1].strip('.')
    candidates = ["father", "mother", "grandfather", "grandmother", "son", "daughter", "grandson", "granddaughter", 
                  "father-in-law", "mother-in-law", "son-in-law", "daughter-in-law", "sister", "brother", "uncle", "aunt"
                  "nephew", "niece", "stepson", "stepdaughter"]
    words = sentence.lower().split(' ')
    words = [word.strip('.').strip(',').strip() for word in words]
    words_in_candidates = [word for word in words if word in candidates]
    answer = words_in_candidates[-1] if words_in_candidates else words[-1]
    return answer

def load_data():
    
    #train_path = "/data/sunwangtao/datasets/reasoning/clutrr/data_emnlp_final/data_089907f8/1.2,1.3_train.csv"
    train_path = "./datasets/clutrr/1.2,1.3_train.csv"

    train_data = pd.read_csv(train_path)[300:500]

    test_path = "./datasets/clutrr/1.2_test.csv"

    test_data = pd.read_csv(test_path)[:100]
    
    from sklearn.utils import shuffle
    from utils import random_seed

    combined_data = pd.concat([train_data, test_data], ignore_index=True)

    shuffled_data = shuffle(combined_data, random_state=random_seed)

    train_data = shuffled_data[:len(train_data)]

    test_data = shuffled_data[len(train_data):]
    
    return train_data, test_data 
