import json

def load_scores(filename):
    with open(filename, 'r') as f:
        scores = json.load(f)
    return scores

dataset = "lets"

expnote_scores = load_scores(f"../exp/{dataset}/chatgpt/expnote/test_scores.json")
disabled_scores = load_scores(f"../exp/{dataset}/chatgpt/expnote_disabled/test_scores.json")

cnt = 0

failed = 0
success = 0
no_need = 0
false = 0

for key in range(100):
    key = str(key)
    if disabled_scores[key] == 0 and expnote_scores[key] == 0:
        failed += 1
    elif disabled_scores[key] == 0 and expnote_scores[key] == 1:
        success += 1
    elif disabled_scores[key] == 1 and expnote_scores[key] == 1:
        no_need += 1
    elif disabled_scores[key] == 1 and expnote_scores[key] == 0:
        false += 1

print(sum(disabled_scores.values()))
print(sum(expnote_scores.values()))

print(f"F=>F: {failed}")
print(f"F=>T: {success}")
print(f"T=>T: {no_need}")
print(f"T=>F: {false}")