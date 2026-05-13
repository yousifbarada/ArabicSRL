import os 
import cv2

data_dir='./data'

noc=28
dataset_Size =350

cap = cv2.VideoCapture(0)
for i in range(noc):
    if not os.path.exists(os.path.join(data_dir,str(i))):
        os.makedirs(os.path.join(data_dir,str(i)))

        print("Collecting data for class {}".format(i))

        done = False
        while True:
            ret,frame = cap.read()
            cv2.putText(frame,'Ready to collect data for class  {} press s to collect'.format(i),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        coutnter=0
        while coutnter<dataset_Size:
            ret,frame = cap.read()
            cv2.imshow('frame',frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(data_dir,str(i),'{}.jpg'.format(coutnter)),frame)
            coutnter+=1
cap.release()
cv2.destroyAllWindows()           