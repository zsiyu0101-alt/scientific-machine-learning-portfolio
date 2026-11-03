import cv2
import numpy as np
import matplotlib.pyplot as plt

def findPendantCurve_Convex(img,numerical=True,k=0.3,img_3ch=None,sampleSize=100):
    
    threshed=cv2.threshold(img,img.mean()+(img.max()-img.mean())*k,255,0)[1]

    #Find the contour with the greatest arc length.
    cnts,hier=cv2.findContours(threshed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    lengths=np.zeros(len(cnts))
    for t,c in enumerate(cnts):
        lengths[t]=cv2.arcLength(c,False)
    Curve=cnts[np.argmax(lengths)]
    
    #Downsample the dense contour from the high-resolution image.
    original_length=len(Curve)
    choice=np.linspace(0,original_length,
                       1000 if original_length>1000 else original_length,
                       endpoint=False,dtype=np.int32)
    #Locate the bottom tip.
    tip_index=np.argmax(Curve[::,0,1])

    #Build a convex hull for convexity-defect detection.
    hull_points=cv2.convexHull(Curve,returnPoints=True)
    hull_indices=cv2.convexHull(Curve,returnPoints=False)
    
    #Select suitable defect extrema as boundary points.
    convexDefects=cv2.convexityDefects(Curve,hull_indices)
    convexDef_length=convexDefects[::,0,1]-convexDefects[::,0,0]
    # (convexDef_indices,)=np.where(convexDef_length>50)
    convexDef_indices=np.argsort(convexDef_length)[-2:]

    left_index,right_index=convexDefects[convexDef_indices,0,2]
    
   
    #Extract the pendant-drop segment of the contour.
    l,r=sorted([left_index,right_index])
    PendantCurve=Curve[l:r+1]
    tip=np.argmin(PendantCurve[::,0,1])

    #Compute the capillary diameter from the distance between boundary points.
    diameter=np.sum(np.square(PendantCurve[0,0,::]-PendantCurve[-1,0,::]))
    diameter=np.sqrt(diameter)
    
    #Move the drop tip to the origin and reverse the vertical axis.
    PendantCurve_numerical=(PendantCurve-PendantCurve[tip]).reshape(-1,2)
    PendantCurve_numerical=-np.array(PendantCurve_numerical,dtype=np.float32)
    # PendantCurve_numerical/=diameter
    
    fig=None
    if img_3ch is not None:
        img_contour=img_3ch.copy()
        img_contour=cv2.drawContours(img_contour,[PendantCurve],-1,(0,255,0),
                (np.max(PendantCurve_numerical[::,0])-np.min(PendantCurve_numerical[::,0]))*0.1)
        b,g,r=cv2.split(img_contour)
        img_contour=cv2.merge([r,g,b])
        fig=plt.figure(figsize=(8,6))
        plt.imshow(img_contour)
        plt.axis('off')
        # fig.savefig('./DetectedCurve')
    
    return PendantCurve_numerical,tip,diameter,fig

def sample(curve,size):
    choices=np.linspace(0,len(curve)-1,size,dtype=np.int32)
    return curve[choices,::]


