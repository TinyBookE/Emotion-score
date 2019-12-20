# Emotion-score

Score six emotions based on natural scenes, such as cityscape.

~~It uses VGG19 simply now. And it may be modified not long after depended on the discussion of our group.~~

Comparing the results of VGG19 and ResNet50. We think that VGG19 performs better in this job. 

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
- utils.py some utils.

# Predict usage
`
python predict.py <-e|--emotion emotion> <-f|--from from_dir> <-t|--to to_dir>
`

## emotion options
1. beautiful
2. boring
3. depressing
4. lively
5. safety
6. wealthy