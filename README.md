#  Person-Re-Identification-Baseline-Based-on-Attention-Block-Neural-Architecture-Search
 This is the Pytorch code of the paper ''Person Re-Identification Baseline Based on Attention Block Neural Architecture Search''.

## Installation
- Install [pytorch](https://pytorch.org/get-started/previous-versions/) with python 3.7, pytorch==1.4.0, torchvision==0.5.0, CUDA==10.1.
- Clone this repository:  
```
git clone https://github.com/Nicholasxin/Person-Re-Identification-Baseline-Based-on-Attention-Block-Neural-Architecture-Search
  
```

## Dataset
- Before run or debug the code, the folder named 'data' should be built and put the ReID public database in it directly. Then, RUN or DEBUG the code.


## Training
- Firstly, run the script train_search.py. The parameters of the four searched attention blocks would be print in the terminal. 
- Secondly, put the obtained 4 parameters in the script '.NAS/genotypes.py' manually.
- Finally, run the script train.py to implement the ReID task by the searched network architecture. 

## Testing
- The test process will run at the same time with training while running the script train.py. 


## Note
- 31/05/2022: code released
