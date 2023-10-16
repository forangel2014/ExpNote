import pandas as pd
import logging
import json
import argparse
from utils import *

task_instruction = """
SYSTEM: Your task is to splice the specified letters together.
"""

expnote_train = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: Splice the 6th letter of "turnip", the 2rd letter of "tablet", and the 4th letter of "coin" together.
ASSISTANT: THINK[The 6th letter of "turnip" is "p", the 2rd letter of "tablet" is "a", and the 4th letter of "coin" is "n".]
SYSTEM: OK.
ASSISTANT: ANSWER[So the spliced letters would be "pan".]
SYSTEM: Your answer is correct.
ASSISTANT: THINK[I should note the spelling of these words for future reference]
SYSTEM: OK.
ASSISTANT: NOTE[turnip]: "turnip" is composed of "t, u, r, n, i, p". So for the word "turnip", the 1st letter is "t", 2rd letter is "u", 3rd letter is "r", 4th letter is "n", 5th letter is "i", and the 6th letter is "p". 
SYSTEM: OK.
ASSISTANT: NOTE[tablet]: "tablet" is composed of "t, a, b, l, e, t". So for the word "tablet", the 1st letter is "t", 2rd letter is "a", 3rd letter is "b", 4th letter is "l", 5th letter is "e", and the 6th letter is "t". 
SYSTEM: OK.
ASSISTANT: NOTE[coin]: "coin" is composed of "c, o, i, n". So for the word "coin", the 1st letter is "c", 2rd letter is "o", 3rd letter is "i", and the 4th letter is "n". 
SYSTEM: OK.
SYSTEM: case 2.
SYSTEM: Splice the 4th letter of "manual", the 3th letter of "represent", and the 4th letter of "official" together.
ASSISTANT: THINK[The 4th letter of "manual" is "u", the 3th letter of "represent" is "r", and the 4th letter of "official" is "i".]
SYSTEM: OK.
ASSISTANT: ANSWER[So the spliced letters would be "uri".]
SYSTEM: Your answer is "uri", however, the correct answer is "upi".
ASSISTANT: THINK[I should note the spelling of these words for future reference]
SYSTEM: OK.
ASSISTANT: NOTE[manual]: "manual" is composed of "m, a, n, u, a, l". So for the word "manual", the 1st letter is "m", 2rd letter is "a", 3rd letter is "n", 4th letter is "u", 5th letter is "a", and the 6th letter is "l".
SYSTEM: OK.
ASSISTANT: NOTE[represent]: "represent" is composed of "r, e, p, r, e, s, e, n, t". So for the word "represent", the 1st letter is "r", 2rd letter is "e", 3rd letter is "p", 4th letter is "r", 5th letter is "e", 6th letter is "s", 7th letter is "e", 8th letter is "n", and the 9th letter is "t".]
SYSTEM: OK.
ASSISTANT: NOTE[official]: "official" is composed of "o, f, f, i, c, i, a, l". So for the word "official", the 1st letter is "o", 2rd letter is "f", 3rd letter is "f", 4th letter is "i", 5th letter is "c", 6th letter is "i", 7th letter is "a", and the 8th letter is "l".]
SYSTEM: OK.
"""

expnote_test_experience = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: Splice the 4th letter of "manual", the 3th letter of "represent", and the 4th letter of "official" together.
SYSTEM: No relevant experience.
ASSISTANT: THINK[Seems there is no relevant experience about this question, I need to rely on myself]
SYSTEM: OK.
ASSISTANT: THINK[The 4th letter of "manual" is "u", the 3th letter of "represent" is "p", and the 4th letter of "official" is "i".]
SYSTEM: OK.
ASSISTANT: ANSWER[So the spliced letters would be "upi".]
SYSTEM: Your answer is correct.
SYSTEM: case 2.
SYSTEM: Splice the 6th letter of "turnip", the 3rd letter of "tablet", and the 4th letter of "coin" together.
SYSTEM: Here are several relevant experience:
SYSTEM: "turnip" is composed of "t, u, r, n, i, p". So for the word "turnip", the 1st letter is "t", 2rd letter is "u", 3rd letter is "r", 4th letter is "n", 5th letter is "i", and the 6th letter is "p". 
SYSTEM: "coin" is composed of "c, o, i, n". So for the word "coin", the 1st letter is "c", 2rd letter is "o", 3rd letter is "i", and the 4th letter is "n".
ASSISTANT: THINK[Based on these experiences, the first letter is "p", the third letter is "n"]
SYSTEM: OK.
ASSISTANT: THINK[And the the 3rd letter of "tablet" is "b", so the second letter is "b"]
SYSTEM: OK.
ASSISTANT: ANSWER[So the spliced letters would be "pbn".]
SYSTEM: Your answer is correct.
"""

expnote_test_case = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: Splice the 4th letter of "manual", the 3th letter of "represent", and the 4th letter of "official" together.
SYSTEM: No relevant experience.
ASSISTANT: THINK[Seems there is no relevant experience about this question, I need to rely on myself]
SYSTEM: OK.
ASSISTANT: THINK[The 4th letter of "manual" is "u", the 3th letter of "represent" is "p", and the 4th letter of "official" is "i".]
SYSTEM: OK.
ASSISTANT: ANSWER[So the spliced letters would be "upi".]
SYSTEM: Your answer is correct.
SYSTEM: case 2.
SYSTEM: Splice the 6th letter of "turnip", the 3rd letter of "tablet", and the 4th letter of "coin" together.
SYSTEM: Here are several relevant experience:
SYSTEM: Splice the 3th letter of "supper", the 1th letter of "center", and the 5rd letter of "turnip" together.
Your answer is "pcp", the correct answer is "pci".
SYSTEM: Splice the 5th letter of "grave", the 4th letter of "dispose", and the 2rd letter of "coin" together.
Your answer is "eoo", the correct answer is "epo".
ASSISTANT: THINK[Based on these experiences, the first letter is "p", the third letter is "n"]
SYSTEM: OK.
ASSISTANT: THINK[And the the 3rd letter of "tablet" is "b", so the second letter is "b"]
SYSTEM: OK.
ASSISTANT: ANSWER[So the spliced letters would be "pbn".]
SYSTEM: Your answer is correct.
"""

few_shot = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: Splice the 3th letter of "possess", the 4th letter of "sometime", and the 5th letter of "butcher" together.
ASSISTANT: seh
SYSTEM: case 2.
SYSTEM: Splice the 6th letter of "turnip", the 2rd letter of "tablet", and the 4th letter of "coin" together.
ASSISTANT: pan
SYSTEM: case 3.
SYSTEM: Splice the 4th letter of "manual", the 3th letter of "represent", and the 4th letter of "official" together.
ASSISTANT: upi
"""

cot = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: Splice the 3th letter of "possess", the 4th letter of "sometime", and the 5th letter of "butcher" together.
ASSISTANT: The 3th letter of "possess" is "s", the 4th letter of "sometime" is "e", and the 5th letter of "butcher" is "h". So the spliced letters would be "seh".
SYSTEM: case 2.
SYSTEM: Splice the 6th letter of "turnip", the 2rd letter of "tablet", and the 4th letter of "coin" together.
ASSISTANT: The 6th letter of "turnip" is "p", the 2rd letter of "tablet" is "a", and the 4th letter of "coin" is "n". So the spliced letters would be "pan".
SYSTEM: case 3.
SYSTEM: Splice the 4th letter of "manual", the 3th letter of "represent", and the 4th letter of "official" together.
ASSISTANT: The 4th letter of "manual" is "u", the 3th letter of "represent" is "p", and the 4th letter of "official" is "i". So the spliced letters would be "upi".
"""

reflexion = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: Splice the 3th letter of "possess", the 4th letter of "sometime", and the 5th letter of "butcher" together.
ASSISTANT: The 3th letter of "possess" is "s", the 4th letter of "sometime" is "e", and the 5th letter of "butcher" is "e". So the spliced letters would be "see".
SYSTEM: Your answer is incorrect.
ASSISTANT: Reflection: I mistakenly thought the 5th letter of "butcher" is "e", I should try again.
SYSTEM: Splice the 3th letter of "possess", the 4th letter of "sometime", and the 5th letter of "butcher" together.
ASSISTANT: The 3th letter of "possess" is "s", the 4th letter of "sometime" is "e", and the 5th letter of "butcher" is "h". So the spliced letters would be "seh".
SYSTEM: case 2.
SYSTEM: Splice the 6th letter of "turnip", the 2rd letter of "tablet", and the 4th letter of "coin" together.
ASSISTANT: The 6th letter of "turnip" is "i", the 2rd letter of "tablet" is "a", and the 4th letter of "coin" is "n". So the spliced letters would be "ian".
SYSTEM: Your answer is incorrect.
ASSISTANT: Reflection: I mistakenly thought the 6th letter of "turnip" is "i", I should try again.
SYSTEM: Splice the 6th letter of "turnip", the 2rd letter of "tablet", and the 4th letter of "coin" together.
ASSISTANT: The 6th letter of "turnip" is "p", the 2rd letter of "tablet" is "a", and the 4th letter of "coin" is "n". So the spliced letters would be "pan".
"""

teachme = """
SYSTEM: Here are several examples.
SYSTEM: case 1.
SYSTEM: Splice the 4th letter of "manual", the 3th letter of "represent", and the 4th letter of "official" together.
SYSTEM: No relevant experience.
ASSISTANT: The 4th letter of "manual" is "u", the 3th letter of "represent" is "p", and the 4th letter of "official" is "i". So the spliced letters would be "upi".
SYSTEM: case 2.
SYSTEM: Splice the 6th letter of "turnip", the 3rd letter of "tablet", and the 4th letter of "coin" together.
SYSTEM: Here are several relevant experience:
SYSTEM: splicing Synonym splice. splicing FormOf splice. polypyrimidine RelatedTo splicing
SYSTEM: letters AtLocation the mail box. エンド マーク Synonym letters. letters ReceivesAction put in letter slots. letters IsA culture
SYSTEM: recombination IsA combining. konbinazio Synonym combining
ASSISTANT: Based on these experiences, the first letter is "p", the third letter is "n", and the the 3rd letter of "tablet" is "b", so the second letter is "b"So the spliced letters would be "pbn".
"""

# reflection = [("THINK[I should note the spelling of these words for future reference]", False),
#               ("NOTE[", True),
#               ("NOTE[", True),
#               ("NOTE[", True)]

reflection = [("", True),
              ("", True),
              ("", True),
              ("", True)]

incorrect = "SYSTEM: Your answer is \"{}\", however, the correct answer is \"{}\".\n"
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
    question, answer = sample.split('\t')
    target = answer.strip()
    return question, target, question

def extract_answer(sentence):
    #response.split(' ')[-1].strip('.')
    words = sentence.lower().strip().strip('.').strip().split(' ')
    answer = words[-1].strip("\"")
    return answer

def load_data():

    lets_path = "./datasets/lets/"

    train_data = open(lets_path + "train.txt").readlines()
    train_data = train_data[:100]
    
    test_data = open(lets_path + "test.txt").readlines()
    test_data = test_data[:100]

    n_train = len(train_data)
    all_data = train_data + test_data
    random.shuffle(all_data)
    train_data = all_data[:n_train]
    test_data = all_data[n_train:]
    
    return train_data, test_data