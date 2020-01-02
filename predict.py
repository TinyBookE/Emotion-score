from Dataset import CustomData
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Model import loadModel
import pandas as pd
import sys, getopt, os



def predict(emotion, from_dir, to_dir):
    custom_data = CustomData(from_dir, isTrain=False)
    dataset = DataLoader(custom_data, shuffle=False)

    df = pd.DataFrame(columns=['filename', 'score'])

    with torch.no_grad():
        model, _, _ = loadModel(emotion, model_type='ResNet', save_dir='./checkpoints/ResNet50', isTrain=False)

        for i, data in enumerate(dataset):
            score, _ = model(Variable(data['img']).cuda())
            df.loc[i] = [data['name'][0], score.cpu().numpy()[0]]
            if (i+1) % 100 == 0:
                print('Score %d pictures'%(i+1))

    print('Score %d pictures'%(len(dataset)))
    to_file = os.path.join(to_dir, '%s_score.csv' % emotion)
    df.to_csv(to_file, index=False)


if __name__ == "__main__":
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

    assert (from_dir is not None and to_dir is not None)

    if emotion is not None:
        print('start scoring %s' % emotion)
        predict(emotion, from_dir, to_dir)
    else:
        emotion = ['beautiful', 'boring', 'depressing', 'lively', 'safety', 'wealthy']
        for e in emotion:
            print('start scoring %s' % e)
            predict(e, from_dir, to_dir)