# create virtual environment
```
conda create -n expnote python=3.8
conda activate expnote
pip install -r requirements.txt
```

# run the main experiments
```
python main.py --dataset clutrr --setting expnote --training true
```

# run the improvement analysis
```
cd scripts
python analysis.py
```