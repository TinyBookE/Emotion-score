import cv2
import os.path
import pandas as pd
import sys, getopt

def split(file, interval = 1):
    file_path, inputfile = os.path.split(file)

    csv_file = os.path.join(file_path, inputfile+'.csv')
    csv_data = pd.read_csv(csv_file, index_col=False)
    out_file = os.path.join(file_path, 'Temp', '{}_{}.png')

    longitude = csv_data.Longitude.values
    Latitude = csv_data.Latitude.values

    vc = cv2.VideoCapture(file)
    FrameRate = vc.get(cv2.CAP_PROP_FPS)
    timeRate = 1/FrameRate

    curFrameTime = 0
    lastFrameTime = -1

    nextPickTime = 0
    count = 0
    totalCount = len(csv_data)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    while rval and count < totalCount:
        if curFrameTime >= nextPickTime and lastFrameTime < nextPickTime:
            out = out_file.format(longitude[count], Latitude[count])
            cv2.imwrite(out, frame)
            nextPickTime += interval
            count += 1

        lastFrameTime = curFrameTime
        curFrameTime += timeRate
        rval, frame = vc.read()

    vc.release()


if __name__ == "__main__":
    assert len(sys.argv) > 1
    argv = sys.argv[1:]
    inputfile = None

    try:
      opts, args = getopt.getopt(argv,"hi:",["ifile="])

    except getopt.GetoptError:
       print('splitVideo.py -i <inputfile>')
       sys.exit(2)

    for opt, arg in opts:
       if opt == '-h':
          print('splitVideo.py -i <inputfile>')
          sys.exit()
       elif opt in ("-i", "--ifile"):
          inputfile = arg

    if inputfile is None:
        print('splitVideo.py -i <inputfile>')
        sys.exit()
    split(inputfile, 1)
