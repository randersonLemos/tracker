import cv2
from datetime import datetime

class MyLog:
    @classmethod
    def myprint(cls, text, camera_id=None):
        owner = cls.__name__
        #print('{}[INFO][{}]{}'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S"), owner, text))        
        if camera_id:
            print('{}[Camera {}][{}]{}'.format(datetime.now().strftime("%m/%d %H:%M:%S"), camera_id, owner, text))        
        else:
            print('{}[{}]{}'.format(datetime.now().strftime("%m/%d %H:%M:%S"), '{:*>15}'.format(owner if len(owner) < 15 else owner[:15]), text))        


    #@classmethod
    #def mywarn(cls, text):
    #    owner = cls.__name__
    #    print('{}[WARN][{}]{}'.format(datetime.now().strftime("%/m/%d/%Y %H:%M:%S"), owner, text))


    @classmethod
    def myimshow(cls, name, img):
        owner = cls.__name__
        name = '[{}]{}'.format(owner, name)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 600,600)
        cv2.imshow(name, img)


    @classmethod
    def mywaitKey(cls, wait=1):
        cv2.waitKey(wait)
