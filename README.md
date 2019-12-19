# Emotion-score

Score six emotions based on natural scenes, such as cityscape.

It uses VGG19 simply now. And it may be modified not long after depended on the discussion of our group. 

# Environment
- python3
- pytorch
- cuda

# Instruction
- Dataset.py: load image from img_dir and label from label_csv
- Model.py: model class.
- RF.py: random forests, the first thoughts. But it is deprecated now.
- train.py: train model.
- test.py: apply the model to the test set.
- predict.py: give the score.

## Not completed