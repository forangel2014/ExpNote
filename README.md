# ExpNote
This is the repository of [ExpNote: Black-box Large Language Models are Better Task Solvers with Experience Notebook](https://arxiv.org/abs/2311.07032) (EMNLP 2023 findings)

## create virtual environment
```
conda create -n expnote python=3.8
conda activate expnote
pip install -r requirements.txt
```

## run the main experiments
```
python main.py --dataset clutrr --setting expnote --training true
```

## run the improvement analysis
```
cd scripts
python analysis.py
```
