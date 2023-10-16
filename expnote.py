import json
import argparse
from utils import filter_exp, concept_net
from lm import WordBasedRetriever, BM25Retriever, DenseRetriever

class ExpNote():
    
    def __init__(self, args):
        self.args = args
        self.content = args.content
        self.clear()
        
        self.prompts_train = "This is training stage, you should first answer the question directly through using ANSWER command. \
Then you should reflect your answer and induce relevant experiences through NOTE command. When you need to organize your thought, use THINK command."
        self.prompts_test = "This is testing stage, you should consider the retrieved experiences and answer the question through using ANSWER command. \
When you need to organize your thought, use THINK command."


    def __call__(self, sentence):
        try:
            if "THINK[" in sentence:
                return True, "OK."
            elif "NOTE[" in sentence and not "MISNOTE" in sentence:
                self.note(sentence)
                return True, "OK."
            elif "RECALL[" in sentence:
                key = sentence.split("]")[0]
                key = key.split("RECALL[")[-1].strip()
                return True, self.recall(key)
            elif "MISNOTE" in sentence:
                if "MISNOTE[" not in sentence:
                    return True, self.get_keys()
                else:
                    key = sentence.split("]")[0]
                    key = key.split("MISNOTE[")[-1].strip()
                    return True, self.retrieve(key)
            elif "ANSWER" in sentence:
                sentence = sentence.split("]")[0]
                sentence = sentence.split("ANSWER[")[-1].strip()
                return False, sentence
            else:
                return True, "Invalid action."
        except:
            return True, "Invalid action."

    def start_case(self, question, target):
        self.case_mem = {"question": question, "target": target, "answer": None, "prompt": None, "score": None, "experience": {}}
        
    def answer_case(self, answer, prompt, score):
        self.case_mem["answer"] = answer
        self.case_mem["prompt"] = prompt
        self.case_mem["score"] = score

    def end_case(self):
        if self.case_mem["answer"] is not None and len(self.case_mem["experience"]) > 0:
            for key, value in self.case_mem["experience"].items():
                experience = {"question": self.case_mem["question"], 
                            "target": self.case_mem["target"], 
                            "answer": self.case_mem["answer"],
                            "keyword": key,
                            "experience": value}
                key2experience = {key:experience}
                question2experience = {self.case_mem["question"]:experience}
                self.key2experience.update(key2experience)
                self.question2experience.update(question2experience)
                
                # if self.recall_method == "retrieve":
                #     self.retriever.encode_new_docs("key", key2experience)
                #     self.retriever.encode_new_docs("question", question2experience)
                
            self.labeled_cases.append({"question": self.case_mem["question"], 
                                        "target": self.case_mem["target"], 
                                        "answer": self.case_mem["answer"],
                                        "prompt": self.case_mem["prompt"]})
    
    def get_keys(self):
        if len(self.key2experience):
            return ", ".join(map(lambda x: "[{}]".format(x), self.key2experience.keys()))
        else:
            return "The MISNOTE is empty."

    def retrieve(self, key, k=5):
        if len(self.key2experience):
            res = self.retriever.retrieve("key", key, k)
            res = ["[{}]".format(kw) for kw in res]
            return ", ".join(res)
        else:
            return "The MISNOTE is empty."        

    def note(self, sentence):
        #self.notes[key] = value
        # if self.content == "experience":
        key, value = sentence.split("]:")
        key = key.split("NOTE[")[-1].strip().lower()
        value = value.strip()
        self.case_mem["experience"][key] = value #TODO 实时更新
        # elif self.content == "case":
        #     key = sentence.split("NOTE[")[-1].strip()
        #     key = key.split("]")[0].strip()
        #     self.case_mem["experience"][key] = 0 #TODO 实时更新

    def recall(self, key):
        if self.content == "experience":
            if key in self.key2experience.keys():
                return self.key2experience[key]["experience"]
            else:
                return "Tag not exist."
        elif self.content == "case":
            if key in self.key2experience.keys():
                return self.key2experience[key]["question"] + "\nYour answer is \"{}\", the correct answer is \"{}\".".format(
                        self.key2experience[key]["answer"], 
                        self.key2experience[key]["target"])
            else:
                return "Tag not exist."
        else:
            raise Exception("Invalid content")

    def auto_recall(self, model, question_text):
        
        if self.args.dataset == "emoji":
            k = 5
        else:
            k = 3
            
        experiences = []
        if not self.args.disabled:
            if self.args.setting == "expnote":
                if len(self.key2experience):
                    keys = self.retriever.retrieve("key", question_text, k)
                    for key in keys:
                        experience = self.recall(key)
                        experiences.append(experience)
            else:
                assert self.args.setting == "teachme"
                experiences = concept_net(model, question_text)[:k]
            
        if len(experiences):
            prompt = "SYSTEM: Here are several possible relevant experience:\n"
            for experience in experiences:
                prompt += f"SYSTEM: {experience}\n"
        else:
            prompt = "SYSTEM: No relevant experience.\n"
                
        return prompt

    def clear(self):
        self.notes = []
        self.key2experience = {}
        self.question2experience = {}
        self.labeled_cases = []
        #self.retriever = DenseRetriever(device=args.device)
        #self.retriever = BM25Retriever()
        self.retriever = WordBasedRetriever(self.args)

    def save(self, filename):
        self.notes = {"key2experience": self.key2experience, 
                      "question2experience": self.question2experience,
                      "labeled_cases": self.labeled_cases}
        with open(filename, 'w') as f:
            json.dump(self.notes, f)
        
    def load(self, filename):
        self.clear()
        with open(filename, 'r') as f:
            self.notes = json.load(f)
            self.key2experience = filter_exp(self.notes["key2experience"], self.args.exp_type)
            self.question2experience = filter_exp(self.notes["question2experience"], self.args.exp_type)
            self.labeled_cases = filter_exp(self.notes["labeled_cases"], self.args.exp_type)
        self.retriever.encode_new_docs("key", self.key2experience)
        self.retriever.encode_new_docs("question", self.question2experience)
        
        print(f"successfully loaded {len(self.key2experience)} experiences")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ExpNote')

    parser.add_argument("--content", type=str, default='experience')
    parser.add_argument("--recall", type=str, default='retrieve')
    args = parser.parse_args()

    # expnote = ExpNote()
    
    # res = expnote(" NOTE[grandparent, grandchild]: if A is the son of B, A has a kid C, then C is also the grandchild of B")
    # print(res)
    # res = expnote(" NOTE[aaa, bbb]:   ccc")
    # print(res)
    # res = expnote(" MISNOTE")
    # print(res)
    # res = expnote(" RECALL[grandparent, grandchild]")
    # print(res)


    expnote = ExpNote(args)
    expnote.load("./exp/letter/chatgpt/expnote/experience/expnote.json")
    expnote("MISNOTE[born]")
    print(expnote.notes)