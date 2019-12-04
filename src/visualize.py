import tensorflow as tf
import cv2
import config as cfg

def logits_to_result(self, model, img):
    '''
    Use model to process a given image and return object detection result.
    Params: Model object and an image.
    Return: Tuple of information of objects. 
            Each is a tuple of bounding box x, y, w, h and corresponding class prediction.
            ((x,y,w,h,c), (x,y,w,h,c), ...)
    '''

    logits = tf.squeeze(model.call(img))        # shape=(7, 7, 30)
    

    return ((1,2,3,4,10), (5,6,7,8,9))


def visualize(self, model, img):
    '''
    Visualize an image with object detextion predictions.
    Params: Model object and an image.
    Return: New image with bounding boxes and class names.
    '''
    
    objects = logits_to_result(model, img)

    for (x, y, w, h, c) in objects:

        # draw a green rectangle to visualize the bounding box
        image = cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)

        # draw class name (a text string)
        class_name = cfg.class_names[c]
        image = cv2.putText( 
            image, class_name, (x,y), font=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA )
        
    cv2.imshow('detection visualization',image)

    # cv2.imwrite('new_image', image)
    # cv2.destroyAllWindows()
