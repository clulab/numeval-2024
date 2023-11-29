# SemEval-2024 Task 7: NumEval Task 3: Numeral-Aware Headline Generation (English)

This task consists of two subtasks:

## Subtask 1: 

Focused on numerical reasoning, models are required to compute the correct number to fill the blank in a news headline. The data sets consists of 4 columns - news, masked headline, calculation and answer. 

### One Step: 

* DistilRoBERTa: sub1_roberta.ipynb - 79.8% accuracy
* T5-base by Michal Pleban: sub1_michau.ipynb - 87.7% accuracy
* T5 by Caleb Zearing: sub1_cz.ipynb - 87.8% accuracy
* LaMini-Flan-T5-783M: sub1_lamini.ipynb - 88.6% accuracy

### Two Steps: 
Step 1: news & masked headlines as inputs and calculations as outputs.
Step 2: calculations as inputs and answers (numbers) as outputs. 

* T5-base by Michal Pleban: sub1_michau_2steps.ipynb - 87.9% accuracy
* T5 by Caleb Zearing: sub1_cz_2steps.ipynb - 88.1% accuracy
* LaMini-Flan-T5-783M: sub1_lamini_2steps_final.ipynb - **90.2%** accuracy

predictions.txt - answers by LaMini-Flan-T5-783M 2 steps

## Subtask 2: 

Models are required to generate an entire headline based on the provided news. 

* T5-base by Michal Pleban: sub2_michal.ipynb (sub2_michal.xlsx: headlines generated)
* T5 by Caleb Zearing: sub2_cz.ipynb (sub2_cz.xlsx: headlines generated)

Here is a poster for this project:
[Poster.pdf](https://github.com/clulab/numeval-2024/files/13503305/Poster.pdf)
