from Dataset import CustomData
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Model import loadModel
import pandas as pd
import sys, getopt, os

argv = sys.argv
assert (len(argv) > 1)

emotion = None
from_dir = None
to_dir = None


opts, args = getopt.getopt(argv[1:], 'e:f:t:', ['emotion=', 'from=', 'to='])
for opt, value in opts:
    if opt in ['-e', '--emotion']:
        emotion = value
    elif opt in ['-f', '--from']:
        from_dir = value
    elif opt in ['-t', '--to']:
        to_dir = value

assert (emotion is not None and from_dir is not None and to_dir is not None)

custom_data = CustomData(from_dir)
dataset = DataLoader(custom_data, shuffle=False)

df = pd.DataFrame(columns=['filename', 'score'])

with torch.no_grad():
    model, _, _ = loadModel(emotion, model_type='VGG', save_dir='./checkpoints/VGG19')

    for i, data in enumerate(dataset):
        score, _ = model(Variable(data['img']).cuda())
        df.loc[i] = [data['name'][0], score.cpu().numpy()[0]]

to_file = os.path.join(to_dir, '%s_score.csv'%emotion)

df.to_csv(to_file, index=False)