# SemEval-2024 Task 7: NumEval Task 3: Numeral-Aware Headline Generation (English)

This task consists of two subtasks:

## Subtask 1: 

Focused on numerical reasoning, models are required to compute the correct number to fill the blank in a news headline. The data sets consists of 4 columns - news, masked headline, calculation and answer. 

### One Step: 

* DistilRoBERTa: sub1_roberta.ipynb - 78.7% accuracy
* T5-base by Michal Pleban: sub1_michau.ipynb - 87.7% accuracy
* T5 by Caleb Zearing: sub1_cz.ipynb - 88.3% accuracy
* LaMini-Flan-T5-783M: sub1_lamini.ipynb - 87.2% accuracy

### Two Steps: 
Step 1: news & masked headlines as inputs and calculations as outputs.
Step 2: calculations as inputs and answers (numbers) as outputs. 

* T5-base by Michal Pleban: sub1_michau_2steps.ipynb - 87.9% accuracy
* T5 by Caleb Zearing: sub1_cz_2steps.ipynb - 88.1% accuracy
* LaMini-Flan-T5-783M: sub1_lamini_step1.ipynb for step 1
                       sub1_lamini_step2.ipynb for step 2 - 89.9% accuracy

## Subtask 2: 

Models are required to generate an entire headline based on the provided news. 

* T5-base by Michal Pleban: sub2_michal.ipynb 
* T5 by Caleb Zearing: sub2_cz.ipynb


![poster](https://github.com/clulab/numeval-2024/assets/101580620/416360d8-8a4f-49e0-a45d-64cc1e56fa9b)
