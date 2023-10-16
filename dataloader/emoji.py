import pandas as pd
import pickle
import numpy as np
from utils import *
np.random.seed(42)

task_instruction = """
SYSTEM: Your task is to select the emoji that best matches the text.
"""

expnote_train = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: When someone stole my bike. \\U0001f616 \\U0001f620 \\U0001f630 \\U0001f97a
ASSISTANT: THINK[The sentiment of this text is angry]
SYSTEM: OK.
ASSISTANT: ANSWER[So the answer is \\U0001f620]
SYSTEM: Your answer is correct.
ASSISTANT: THINK[I should note the meaning of this emoji for future reference]
SYSTEM: OK.
ASSISTANT: NOTE[\\U0001f620]: \\U0001f620 indicates a sentiment of anger
SYSTEM: case 2.
SYSTEM: When I saw the list of books to read for an acccouting course, I thought "" Oh God how will I ever manaage it! \\U0001f616 \\U0001f631 \\U0001f625 \\U0001f92c
ASSISTANT: THINK[The sentiment of this text is sadness]
SYSTEM: OK.
ASSISTANT: ANSWER[So the answer is \\U0001f631]
SYSTEM: Your answer is \\U0001f631, however, the correct answer is \\U0001f625.
ASSISTANT: THINK[I should note the meaning of this emoji for future reference]
SYSTEM: OK.
ASSISTANT: NOTE[\\U0001f625]: \\U0001f625 indicates a sentiment of sadness
SYSTEM: case 3.
SYSTEM: When I dropped a bottle in the state liquor store and everyboby started laughing. \\U0001f97a \\U0001f922 \\U0001f623 \\U0001f630
ASSISTANT: The sentiment of this text is shame, so the answer is \\U0001f97a
ASSISTANT: THINK[The sentiment of this text is shame]
SYSTEM: OK.
ASSISTANT: ANSWER[So the answer is \\U0001f630]
SYSTEM: Your answer is \\U0001f630, however, the correct answer is \\U0001f97a.
ASSISTANT: THINK[I should note the meaning of this emoji for future reference]
SYSTEM: OK.
ASSISTANT: NOTE[\\U0001f97a]: \\U0001f97a indicates a sentiment of shame
"""


#SYSTEM: This is testing stage, need to think whether to refer to the given experience and finally answer the question correctly, here are several examples.

expnote_test_experience = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: When someone stole my bike. \\U0001f616 \\U0001f620 \\U0001f630 \\U0001f97a
SYSTEM: Here are several possible relevant experience:
SYSTEM: \\U0001f616 indicates a sentiment of guilt
SYSTEM: \\U0001f620 indicates a sentiment of anger
SYSTEM: \\U0001f630 indicates a sentiment of fear
SYSTEM: \\U0001f97a indicates a sentiment of shame
ASSISTANT: THINK[The sentiment of this text is angry. Based on these experiences, the answer is \\U0001f620]
SYSTEM: OK.
ASSISTANT: ANSWER[So the answer is \\U0001f620]
SYSTEM: case 2.
SYSTEM: When I saw the list of books to read for an acccouting course, I thought "" Oh God how will I ever manaage it! \\U0001f616 \\U0001f631 \\U0001f625 \\U0001f92c
SYSTEM: Here are several possible relevant experience:
SYSTEM: \\U0001f616 indicates a sentiment of guilt
SYSTEM: \\U0001f631 indicates a sentiment of fear
SYSTEM: \\U0001f625 indicates a sentiment of sadness
SYSTEM: \\U0001f92c indicates a sentiment of anger
ASSISTANT: THINK[The sentiment of this text is sadness. Based on these experiences, the answer is \\U0001f625]
SYSTEM: OK.
ASSISTANT: ANSWER[So the answer is \\U0001f625]
SYSTEM: case 3.
SYSTEM: When I dropped a bottle in the state liquor store and everyboby started laughing. \\U0001f97a \\U0001f922 \\U0001f623 \\U0001f630
SYSTEM: Here are several possible relevant experience:
SYSTEM: \\U0001f97a indicates a sentiment of shame
SYSTEM: \\U0001f922 indicates a sentiment of disgust
SYSTEM: \\U0001f623 indicates a sentiment of guilt
SYSTEM: \\U0001f630 indicates a sentiment of fear
ASSISTANT: THINK[The sentiment of this text is shame. Based on these experiences, the answer is \\U0001f97a]
SYSTEM: OK.
ASSISTANT: ANSWER[So the answer is \\U0001f97a]
"""

expnote_test_case = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: When someone stole my bike. \\U0001f616 \\U0001f620 \\U0001f630 \\U0001f97a
SYSTEM: Here are several possible relevant experience:
SYSTEM: I hapened to overhear something which I was not meant to hear. \\U0001f616 \\U0001f600 \\U0001f622 \\U0001f621
Your answer is "\\U0001f622", the correct answer is "\\U0001f616".
SYSTEM: Nearly caught masturbating. \\U0001f97a \\U0001f61e \\U0001f92e \\U0001f604
Your answer is "\\U0001f61e", the correct answer is "\\U0001f97a".
SYSTEM: At work, when an elderly man complained unjustifiably about me and distrusted me. \\U0001f97a \\U0001f620 \\U0001f630 \\U0001f92e
Your answer is "\\U0001f620", the correct answer is "\\U0001f620".
SYSTEM: When I flew from Athens to Copenhagen after a holiday.  The plane was caught in a storm and it began to shake and I thouhght that we were going to crash. \\U0001f620 \\U0001f97a \\U0001f630 \\U0001f616
Your answer is "\\U0001f630", the correct answer is "\\U0001f630".
ASSISTANT: THINK[The sentiment of this text is angry. Based on these experiences, the answer is \\U0001f620]
SYSTEM: OK.
ASSISTANT: ANSWER[So the answer is \\U0001f620]
SYSTEM: case 2.
SYSTEM: When I saw the list of books to read for an acccouting course, I thought "" Oh God how will I ever manaage it! \\U0001f616 \\U0001f631 \\U0001f625 \\U0001f92c
SYSTEM: Here are several possible relevant experience:
SYSTEM: I hapened to overhear something which I was not meant to hear. \\U0001f616 \\U0001f600 \\U0001f622 \\U0001f621
Your answer is "\\U0001f622", the correct answer is "\\U0001f616".
SYSTEM: A case of unrequited love. \\U0001f625 \\U0001f61e \\U0001f621 \\U0001f612
Your answer is "\\U0001f625", the correct answer is "\\U0001f625".
SYSTEM: At my Summer job, nobody looked after me in particular and I had to learn all on my own. \\U0001f92c \\U0001f62d \\U0001f92e \\U0001f633
Your answer is "\\U0001f92c", the correct answer is "\\U0001f92c".
SYSTEM: When I climbed up a tree to pick apples.  The angle of the ladder I was on did not enable me to get high enough.  This implied that the ladder was not very stable. \\U0001f616 \\U0001f633 \\U0001f631 \\U0001f92c
Your answer is "\\U0001f620", the correct answer is "\\U0001f631".
ASSISTANT: THINK[The sentiment of this text is sadness. Based on these experiences, the answer is \\U0001f625]
SYSTEM: OK.
ASSISTANT: ANSWER[So the answer is \\U0001f625]
SYSTEM: case 3.
SYSTEM: When I dropped a bottle in the state liquor store and everyboby started laughing. \\U0001f97a \\U0001f922 \\U0001f623 \\U0001f630
SYSTEM: Here are several possible relevant experience:
SYSTEM: Nearly caught masturbating. \\U0001f97a \\U0001f61e \\U0001f92e \\U0001f604
Your answer is "\\U0001f61e", the correct answer is "\\U0001f97a".
SYSTEM: I do not help out enough at home. \\U0001f623 \\U0001f604 \\U0001f630 \\U0001f92c
Your answer is "\\U0001f623", the correct answer is "\\U0001f623".
SYSTEM: When I flew from Athens to Copenhagen after a holiday.  The plane was caught in a storm and it began to shake and I thouhght that we were going to crash. \\U0001f620 \\U0001f97a \\U0001f630 \\U0001f616
Your answer is "\\U0001f630", the correct answer is "\\U0001f630".
ASSISTANT: THINK[The sentiment of this text is shame. Based on these experiences, the answer is \\U0001f97a]
SYSTEM: OK.
ASSISTANT: ANSWER[So the answer is \\U0001f97a]
"""

few_shot = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: When someone stole my bike. \\U0001f616 \\U0001f620 \\U0001f630 \\U0001f97a
ASSISTANT: \\U0001f620
SYSTEM: case 2.
SYSTEM: When I saw the list of books to read for an acccouting course, I thought "" Oh God how will I ever manaage it! \\U0001f616 \\U0001f631 \\U0001f625 \\U0001f92c
ASSISTANT: \\U0001f625
SYSTEM: case 3.
SYSTEM: When I dropped a bottle in the state liquor store and everyboby started laughing. \\U0001f97a \\U0001f922 \\U0001f623 \\U0001f630
ASSISTANT: \\U0001f97a
"""

cot = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: When someone stole my bike. \\U0001f616 \\U0001f620 \\U0001f630 \\U0001f97a
ASSISTANT: The sentiment of this text is angry, so the answer is \\U0001f620
SYSTEM: case 2.
SYSTEM: When I saw the list of books to read for an acccouting course, I thought "" Oh God how will I ever manaage it! \\U0001f616 \\U0001f631 \\U0001f625 \\U0001f92c
ASSISTANT: The sentiment of this text is sadness, so the answer is \\U0001f625
SYSTEM: case 3.
SYSTEM: When I dropped a bottle in the state liquor store and everyboby started laughing. \\U0001f97a \\U0001f922 \\U0001f623 \\U0001f630
ASSISTANT: The sentiment of this text is shame, so the answer is \\U0001f97a
"""

reflexion = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: When someone stole my bike. \\U0001f616 \\U0001f620 \\U0001f630 \\U0001f97a
ASSISTANT: The sentiment of this text is angry, so the answer is \\U0001f616
SYSTEM: Your answer is incorrect.
ASSISTANT: Reflection: I mistakenly thought \\U0001f616 expresses angry, I should try again.
SYSTEM: When someone stole my bike. \\U0001f616 \\U0001f620 \\U0001f630 \\U0001f97a
ASSISTANT: The sentiment of this text is angry, so the answer is \\U0001f620
SYSTEM: case 2.
SYSTEM: When I saw the list of books to read for an acccouting course, I thought "" Oh God how will I ever manaage it! \\U0001f616 \\U0001f631 \\U0001f625 \\U0001f92c
ASSISTANT: The sentiment of this text is sadness, so the answer is \\U0001f92c
SYSTEM: Your answer is incorrect.
ASSISTANT: Reflection: I mistakenly thought \\U0001f92c expresses sadness, I should try again.
SYSTEM: When I saw the list of books to read for an acccouting course, I thought "" Oh God how will I ever manaage it! \\U0001f616 \\U0001f631 \\U0001f625 \\U0001f92c
ASSISTANT: The sentiment of this text is sadness, so the answer is \\U0001f625
"""

teachme = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: When someone stole my bike. \\U0001f616 \\U0001f620 \\U0001f630 \\U0001f97a
SYSTEM: Here are several possible relevant experience:
SYSTEM: 表情包 RelatedTo emoji. emoji Synonym פרצופון. emoji SimilarTo emoticon
SYSTEM: a text IsA a book. writing RelatedTo text. a text AtLocation a library. a text UsedFor study. テキス Synonym text. word order PartOf text
SYSTEM: bessó Synonym matching. matching SimilarTo matched
ASSISTANT: The sentiment of this text is angry. Based on these experiences, the answer is \\U0001f620
SYSTEM: case 2.
SYSTEM: When I saw the list of books to read for an acccouting course, I thought "" Oh God how will I ever manaage it! \\U0001f616 \\U0001f631 \\U0001f625 \\U0001f92c
SYSTEM: Here are several possible relevant experience:
SYSTEM: Here are several possible relevant experience:
SYSTEM: 表情包 RelatedTo emoji. emoji Synonym פרצופון. emoji SimilarTo emoticon
SYSTEM: a text IsA a book. writing RelatedTo text. a text AtLocation a library. a text UsedFor study. テキス Synonym text. word order PartOf text
SYSTEM: odabir RelatedTo selection. 撰 Synonym selection
SYSTEM: bessó Synonym matching. matching SimilarTo matched
ASSISTANT: The sentiment of this text is sadness. Based on these experiences, the answer is \\U0001f625
"""

reflection = [("THINK[I should note the meaning of this emoji for future reference]", False),
              ("NOTE[", True),
              ("END", False),
              ("", True)]

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
    target = sample["target"]
    question = text + " " + " ".join(sample["candidates"])
    return question, target, question


emotions = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]
emojis = {
    "joy": ["\\U0001f600", "\\U0001f604", "\\U0001f60a"],
    "fear": ["\\U0001f631", "\\U0001f628", "\\U0001f630"],
    "anger": ["\\U0001f620", "\\U0001f92c", "\\U0001f621"],
    "sadness": ["\\U0001f622", "\\U0001f625", "\\U0001f62d"],
    "disgust": ["\\U0001f922", "\\U0001f92e", "\\U0001f612"],
    "shame": ["\\U0001f633", "\\U0001f97a", "\\U0001f614"],
    "guilt": ["\\U0001f61e", "\\U0001f616", "\\U0001f623"]
}

def extract_answer(sentence):
    #response.split(' ')[-1].strip('.')
    candidates = list(emojis.values())
    candidates = list(sum((sublist for sublist in candidates), []))
    words = sentence.split(' ')
    words = [word.strip('.').strip(',').strip() for word in words]
    words_in_candidates = [word for word in words if word in candidates]
    answer = words_in_candidates[-1] if words_in_candidates else words[-1]
    return answer

def filter_data(data):
    # 将标签从one-hot形式转化为整型数
    labels = [np.argmax(info[b'label']) for info in data[b'info']]
    # 将数据转化为列表，每个元素是一个字典
    filtered_list = []
    for i in range(len(data[b'texts'])):
        # 获取text和target
        text = data[b'texts'][i]
        target = labels[i]
        # 随机选取3个候选标签
        candidates = [target]
        while len(candidates) < 4:
            candidate = np.random.randint(0, 7)
            if candidate != target and candidate not in candidates:
                candidates.append(candidate)
        # 将数据添加到列表中
        filtered_list.append({'text': text, 'target': target, 'candidates': candidates})
    # 划分训练集和测试集
    n_class = 7
    n_train = 100
    n_test = 100
    train_list = []
    test_list = []
    train_count = [0] * n_class
    test_count = [0] * n_class
    num_each_class_train = np.ceil(n_train / n_class)
    num_each_class_test = np.ceil(n_test / n_class)
    for item in filtered_list:
        label = item["target"] - 1
        if train_count[label] < num_each_class_train:
            train_list.append(item)
            train_count[label] += 1
        elif test_count[label] < num_each_class_test:
            test_list.append(item)
            test_count[label] += 1
    return train_list[:n_train], test_list[:n_test]

def mapping(data):
    mapped_list = []
    for item in data:
        # 随机选取3个emoji
        target_emoji = np.random.choice(emojis[emotions[item['target']]], 1)[0]
        candidate_emojis = [target_emoji]
        for candidate in item['candidates']:
            if candidate != item['target']:
                candidate_emoji = np.random.choice(emojis[emotions[candidate]], 1)[0]
                candidate_emojis.append(candidate_emoji)
        # 将映射后的数据添加到新的列表中
        np.random.shuffle(candidate_emojis)
        mapped_item = {
            "text": item["text"],
            "target": target_emoji,
            "candidates": candidate_emojis,
        }
        mapped_list.append(mapped_item)
    return mapped_list

def load_data():
    
    emoji_path = "./datasets/emoji/raw.pickle"

    f = open(emoji_path, "rb")
    data = pickle.load(f, encoding='bytes')
    
    train_data, test_data = filter_data(data)
    
    train_data = mapping(train_data)
    test_data = mapping(test_data)

    n_train = len(train_data)
    all_data = train_data + test_data
    random.shuffle(all_data)
    train_data = all_data[:n_train]
    test_data = all_data[n_train:]
    
    return train_data, test_data