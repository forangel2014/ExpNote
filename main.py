import openai
import argparse
from utils import *
from tqdm import tqdm
from lm import LLM
from expnote import ExpNote

format_prompt = """
SYSTEM: You are a smart assistant and You need to complete the specified task by talking to the SYSTEM. \
The dialog between you and SYSTEM will be shown in the form "ASSISTANT:" and "SYSTEM:".
"""

def load_dataset(dataset):
    if dataset == "clutrr":
        from dataloader.clutrr import prompts, load_data, fetch, extract_answer
    elif dataset == "clutrr3":
        from dataloader.clutrr3 import prompts, load_data, fetch, extract_answer
    elif dataset == "lets":
        from dataloader.lets import prompts, load_data, fetch, extract_answer
    elif dataset == "word_sorting":
        from dataloader.word_sorting import prompts, load_data, fetch, extract_answer
    elif dataset == "mets":
        from dataloader.mets import prompts, load_data, fetch, extract_answer
    elif dataset == "emoji":
        from dataloader.emoji import prompts, load_data, fetch, extract_answer
    else:
        raise Exception("Unknown dataset")
    train_data, test_data = load_data()
    return prompts, train_data, test_data, fetch, extract_answer

def load_environment(dataset):
    if dataset == "webshop":
        from dataloader.webshop import prompts, env
    elif dataset == "alfworld":
        from dataloader.alfworld_env_wrapper import prompts, env
        
    return prompts, env

def run_task(args, exp_path, data, model, expnote, prompts, fetch, extract_answer, mode="test"):
    
    logger = set_logger(exp_path, mode)
    
    n = len(data)
    scores = {}
    i = 0
    while i < n:
        try:
            if args.setting == "generate-few-shot":
                init_prompt = ""
            elif args.setting == "zero-shot":
                init_prompt = format_prompt + prompts["task_instruction"]
            elif args.setting == "few-shot":
                init_prompt = format_prompt + prompts["task_instruction"] + prompts["few_shot"]
            elif args.setting == "cot" or args.setting == "cot-sc":
                init_prompt = format_prompt + prompts["task_instruction"] + prompts["cot"]
            elif args.setting == "reflexion":
                init_prompt = format_prompt + prompts["task_instruction"] + prompts["reflexion"]
            elif args.setting == "teachme":
                init_prompt = format_prompt + prompts["task_instruction"] + prompts["teachme"]
            elif args.setting == "expnote":
                if mode == "train":
                    init_prompt = format_prompt + prompts["task_instruction"] + expnote.prompts_train + prompts["expnote_train"]
                elif mode == "test":
                    if args.content == "experience":
                        init_prompt = format_prompt + prompts["task_instruction"] + expnote.prompts_test + prompts["expnote_test_experience"]
                    elif args.content == "case":
                        init_prompt = format_prompt + prompts["task_instruction"] + expnote.prompts_test + prompts["expnote_test_case"]
            else:
                raise Exception("Invalid setting")
            
            init_prompt += """\nSYSTEM: Now here is the question:\n"""
            prompt = init_prompt
                        
            question, target, text = fetch(data, i)
                
            expnote.start_case(question, target)
            done = False
            prompt += "SYSTEM: " + question + '\n'

            if args.setting == "expnote" and mode == "test":
                auto_prompt = expnote.auto_recall(model, text)
                prompt += auto_prompt
            elif args.setting == "teachme":
                auto_prompt = expnote.auto_recall(model, prompts["task_instruction"][9:] + text)
                prompt += auto_prompt
            
            prompt += "ASSISTANT: "
            
            if args.setting == "generate-few-shot":
                prompt += target
                logger.info(prompt)
                continue
            
            step = 0
            retry = True
            
            while not done and step < 10:
                
                if args.setting =="cot-sc":
                    response = []
                    outputs = ""
                    for k in range(5):
                        cot_response = model(prompt, stop=['\n'], temperature=1)
                        response.append(cot_response)
                        outputs += f"{k+1}. {cot_response}\n"
                    prompt += outputs
                else:
                    response = model(prompt, stop=['\n'])
                    prompt += response + '\n'
                    
                if args.setting == "expnote":
                    is_expnote_command, expnote_response = expnote(response)
                else:
                    is_expnote_command, expnote_response = False, response
                    
                if is_expnote_command:
                    prompt += "SYSTEM: " + expnote_response + '\n'
                    prompt += "ASSISTANT: "
                else:
                    if args.setting =="cot-sc":
                        answers = [extract_answer(cot_response) for cot_response in expnote_response]
                        answer = majority_vote(answers)
                    else:
                        answer = extract_answer(expnote_response)
                    case_prompt = prompt[len(init_prompt):]
                    score = int(answer.lower() == target.lower())
                    expnote.answer_case(answer, case_prompt, score)
                    scores.update({i: score})
                    if score:
                        prompt += prompts["correct"]
                        prompt += "ASSISTANT: "
                    else:
                        if args.setting == "reflexion" and retry:
                            prompt += "SYSTEM: Your answer is incorrect.\n"
                            prompt += "ASSISTANT: "
                            response = model(prompt, stop=['\n'])
                            prompt += response + '\n'
                            prompt += "SYSTEM: " + question + '\n'
                            retry = False
                            continue
                        else:
                            prompt += prompts["incorrect"].format(answer, target)
                            prompt += "ASSISTANT: "
                    if args.setting == "expnote" and mode == "train":
                        for j in range(args.actions_reflect):
                            reflection_prompt, model_generate = prompts["reflection"][j]
                            if model_generate:
                                response = model(prompt + reflection_prompt, stop=['\n'])
                                response = reflection_prompt + response
                            else:
                                response = reflection_prompt
                            prompt += response + '\n'
                            prompt += "SYSTEM: "
                            flag, expnote_response = expnote(response)
                            if flag and expnote_response != "Invalid action.":
                                prompt += expnote_response + '\n'
                                prompt += "ASSISTANT: "
                            else:
                                break
                    done = True
                step += 1
            
            i += 1
            tqdm.write(f"Progress: {i}/{n}")
            expnote.end_case()
            logger.info(prompt)
        except (TimeoutError, openai.error.OpenAIError) as e:
            print(e)
            #scores.update({i: -1})
            continue

        if args.record:
            if i % 25 == 0 and i > 0:
                if args.setting == "expnote" and mode == "train":
                    expnote.save(exp_path + f"expnote_{i}.json")
    
    save_scores(exp_path + f"{mode}_scores.json", scores)
    info = evaluate(scores)
    print(info)
    logger.info(info)
    if args.setting == "expnote" and mode == "train":
        expnote.save(exp_path + "expnote.json")

def main(args):

    exp_path = generate_exp_path(args)
    mkdir(exp_path)
    model = LLM(model=args.model, device=args.device)
    expnote = ExpNote(args)
    
    prompts, train_data, test_data, fetch, extract_answer = load_dataset(args.dataset)

    if args.setting == "expnote":
        if args.training:
            run_task(args, exp_path, train_data, model, expnote, prompts, fetch, extract_answer, mode="train")
        expnote.load(exp_path + "expnote.json")
    if args.setting == "generate-few-shot":
        run_task(args, exp_path, train_data, model, expnote, prompts, fetch, extract_answer, mode="train")
    else:
        run_task(args, exp_path, test_data, model, expnote, prompts, fetch, extract_answer, mode="test")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ExpNote')
    
    parser.add_argument("--dataset", type=str, default='lets', help="dataset")
    parser.add_argument("--model", type=str, default='llama2', help="name of LLM")
    parser.add_argument("--setting", type=str, default='expnote', help="expnote or other baselines")
    parser.add_argument("--content", type=str, default='experience', help="content of experience. experience or case")
    parser.add_argument("--exp_type", type=str, default='all', help="type of experience. all, positive or negative")
    parser.add_argument("--device", type=int, default=6)
    parser.add_argument("--actions_reflect", type=int, default=4, help="number of extra actions to reflect")
    parser.add_argument("--training", type=lambda x: x.lower() == 'true', default='true', help="whether to train the expnote")
    parser.add_argument("--disabled", type=lambda x: x.lower() == 'true', default='false', help="whether to disable the expnote")
    parser.add_argument("--record", type=lambda x: x.lower() == 'true', default='false', help="whether to record the training curve")

    args = parser.parse_args()
    print(args)

    main(args)