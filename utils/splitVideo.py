import cv2
import os.path
import pandas as pd
import sys,getopt

def main(argv):
   inputfile = ''
   outputfile = ''
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
   csvfile=inputfile+'.csv'
   csv_data=pd.read_csv(csvfile, index_col = False)
   longitude=csv_data.Longitude.values
   Latitude=csv_data.Latitude.values
   vc=cv2.VideoCapture(inputfile)
   FrameRate=vc.get(5)
   timeF=round(FrameRate)
   count=0
   allcount=0
   if vc.isOpened():
       rval,frame=vc.read()
   else:
       rval=False
   while rval:
       if(allcount%timeF==0):
           cv2.imwrite('result/'+str(longitude[count])+'_'+str(Latitude[count])+'.png',frame)
           count=count+1
       val,frame=vc.read()
       allcount=allcount+1
   vc.release()
    
        


if __name__ == "__main__":
   main(sys.argv[1:])
    
