'''
Created on Apr 1, 2015

@author: kena
'''
####################################################################
#
# Python module to perform SIFT_orig feature detection. this is
# a simple implementation of the SIFT_orig algorithm (Lowe 2004) 
#
####################################################################

import cv2
import numpy as np
#from cv2  import INTER_LINEAR


#===========================Global Variables and Parameters=============================

iDistMin = 0.5
sigmaMin = 0.50 

stackSpace = 3
numOctaves = 3
CTreshold = 0.0315 # This is reduced to prevent too many points being discarded on contrast
curveRatio = 5    # Edgeness ratio used to discard points on edges
lbda = 1.5         #lambda, gaussian window size 

#---------------------------descriptor parameters------------------------------------------
dWindow = 16
numBins = 36
numHist = 4
numHBins = 8
lbdaDesc = 8


#----------------------------scale space structures----------------------------------------
oStack = np.empty([numOctaves,stackSpace+3], dtype=object)    #structure for holding the octaves of stacks
diffOfGauss = np.empty([numOctaves,stackSpace+2], dtype=object) #structure for holding the difference of Gaussians

#----------------------------key points structures-----------------------------------------
#structure for holding the keypoints at different scales
kPoints = np.empty([1,4],dtype=np.float)
#np.delete(kPoints,0)
ipolKPoints = np.empty([1,8],dtype = np.float)

mStack = np.empty([numOctaves], dtype = object)
orStack = np.empty([numOctaves], dtype = object)

finKPoints = np.empty([1,8],dtype = np.float)
# convKeyPoints = np.
#==========================================================================================


def getImage(url): 
    try:
        imageRead = cv2.imread(url,cv2.CV_LOAD_IMAGE_GRAYSCALE)
        image = cv2.resize(imageRead,(int(imageRead.shape[0]/0.5),int(imageRead.shape[1]/0.5)),0,0,INTER_LINEAR)
        initImage = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        return initImage, imageRead 
    except:
        return None, None 

def GBlur(image, sigma):
    outputArr = cv2.GaussianBlur(image,(0,0),sigma)
    return outputArr       

def resample(image):
    W = image.shape[1]; W2 = int(W/2)
    H = image.shape[0]; H2 = int(H/2)
    output = np.empty([H2,W2])
    for i in range(H2):
        for j in range(W2):
            output[int(i),int(j)] = image[i*2,j*2]
    return output
  
def BuildOctaves(initImage):   
    #compute the first octave. The initial image is resized and Interpolated by a factor of 2   
    iDist = iDistMin
    
    oStack[0,0] = initImage
    for j in range(1,stackSpace+3):
        k = (2.0**(float(j)/float(stackSpace)))
        sigma = (iDist/iDistMin)*sigmaMin*k
        oStack[0,j] = GBlur(oStack[0,0], sigma)
       
    #build other octaves
    for i in range(1,numOctaves): 
        iDist = iDistMin*(2**i)
        oStack[i,0] = resample(oStack[i-1,oStack[i-1].shape[0]-3]) 
        for j in range(1,stackSpace+3):
            k = (2**(float(j)/float(stackSpace)))
            sigma = (iDist/iDistMin)*sigmaMin*k
            oStack[i,j] = GBlur(oStack[i,0], sigma)
                           
def DiffOfGauss():
    for i in range(numOctaves):
        for j in range(stackSpace+2):
            diffOfGauss[i,j] = oStack[i,j+1] - oStack[i,j]

def LocalExtrema():
    global kPoints
    for a in range(numOctaves):
        for b in range(2,stackSpace):
            #find the keypoints
            for i in range(1,diffOfGauss[a,0].shape[0]-1):
                for j in range(1,diffOfGauss[a,0].shape[1]-1):
                    #is a local minimum
                    if (((diffOfGauss[a,b][i,j] < diffOfGauss[a,b][i-1,j-1]) and 
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b][i-1,j]) and 
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b][i-1,j+1]) and
                     
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b][i+1,j-1]) and 
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b][i+1,j]) and 
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b][i+1,j+1]) and
                     
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b][i,j-1]) and 
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b][i,j+1]) and
                    
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b-1][i-1,j-1]) and 
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b-1][i-1,j]) and 
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b-1][i-1,j+1]) and
                      
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b-1][i+1,j-1]) and 
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b-1][i+1,j]) and 
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b-1][i+1,j+1]) and
                      
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b-1][i,j-1]) and
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b-1][i,j]) and
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b-1][i,j+1]) and
                    
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b+1][i-1,j-1]) and 
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b+1][i-1,j]) and 
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b+1][i-1,j+1]) and
                     
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b+1][i+1,j-1]) and 
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b+1][i+1,j]) and 
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b+1][i+1,j+1]) and
                     
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b+1][i,j-1]) and
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b+1][i,j]) and
                         (diffOfGauss[a,b][i,j] < diffOfGauss[a,b+1][i,j+1])) or
                    
                    #is a local maximum
                        ((diffOfGauss[a,b][i,j] > diffOfGauss[a,b][i-1,j-1]) and 
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b][i-1,j]) and 
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b][i-1,j+1]) and
                     
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b][i+1,j-1]) and 
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b][i+1,j]) and 
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b][i+1,j+1]) and
                     
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b][i,j-1]) and 
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b][i,j+1]) and
                                            
                        (diffOfGauss[a,b][i,j] > diffOfGauss[a,b-1][i-1,j-1]) and 
                        (diffOfGauss[a,b][i,j] > diffOfGauss[a,b-1][i-1,j]) and 
                        (diffOfGauss[a,b][i,j] > diffOfGauss[a,b-1][i-1,j+1]) and
                      
                        (diffOfGauss[a,b][i,j] > diffOfGauss[a,b-1][i+1,j-1]) and 
                        (diffOfGauss[a,b][i,j] > diffOfGauss[a,b-1][i+1,j]) and 
                        (diffOfGauss[a,b][i,j] > diffOfGauss[a,b-1][i+1,j+1]) and
                      
                        (diffOfGauss[a,b][i,j] > diffOfGauss[a,b-1][i,j-1]) and
                        (diffOfGauss[a,b][i,j] > diffOfGauss[a,b-1][i,j]) and
                        (diffOfGauss[a,b][i,j] > diffOfGauss[a,b-1][i,j+1]) and
                    
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b+1][i-1,j-1]) and 
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b+1][i-1,j]) and 
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b+1][i-1,j+1]) and
                     
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b+1][i+1,j-1]) and 
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b+1][i+1,j]) and 
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b+1][i+1,j+1]) and
                     
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b+1][i,j-1]) and
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b+1][i,j]) and
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b+1][i,j+1]))): #the point is a max or a min
                    
                        kPoints = np.insert(kPoints, 0, [a,b,i,j]) #(kPoints,1,[a,b,i,j],axis=0)
                        
    #delete the last record. this is needed to remove the initial values in the array
    for i in range(0,4):
        kPoints = np.delete(kPoints, kPoints.shape[0]-1)   
                      
def KeyPointsFilter():
    
    global ipolKPoints 
    xCap = np.empty([1,2])  
    #validpoint(a,b,i,j,x,y,sigma,tetha)
 
    #interpolate the points
    for inc in range(0,kPoints.shape[0],4):
        # 8 dimensional array
        candidatePoint = np.array([kPoints[inc],kPoints[inc+1],kPoints[inc+2],kPoints[inc+3],0.0,0.0,0.0,0.0],dtype = float)
        success = False 
        tries = 0
        while (tries < 5): 
            a = candidatePoint[0]; b = candidatePoint[1]
            i = candidatePoint[2]; j = candidatePoint[3]
           
            if (not(((0 < i ) and (i < diffOfGauss[a,2].shape[0]-1)) and ((0 < j ) and (j < diffOfGauss[a,1].shape[1]-1)))):
                break #point out of bounds
            
            gMatrix = np.matrix([((diffOfGauss[a,b][i+1,j])-(diffOfGauss[a,b][i-1,j]))/2, #column difference
                                    ((diffOfGauss[a,b][i,j+1])-(diffOfGauss[a,b][i,j-1]))/2]) #row difference

            HMatrix = np.matrix([[(diffOfGauss[a,b][i+1,j])+(diffOfGauss[a,b][i-1,j])-(2*diffOfGauss[a,b][i,j]),((diffOfGauss[a,b][i+1,j+1])-(diffOfGauss[a,b][i+1,j-1])-(diffOfGauss[a,b][i-1,j+1])+(diffOfGauss[a,b][i-1,j-1]))/4],
                                 [((diffOfGauss[a,b][i+1,j+1])-(diffOfGauss[a,b][i+1,j-1])-(diffOfGauss[a,b][i-1,j+1])+(diffOfGauss[a,b][i-1,j-1]))/4,(diffOfGauss[a,b][i,j+1])+(diffOfGauss[a,b][i,j-1])-(2*diffOfGauss[a,b][i,j])]])
                
            Trace = np.trace(HMatrix)
            Det = np.linalg.det(HMatrix)
            Harris = (Trace**2)/Det
            xCap = gMatrix.dot(-(np.linalg.inv(HMatrix)))
            
            if (abs(xCap[0,0]) < 0.6 and abs(xCap[0,1]) < 0.6 ): #only test for difference of x,y
                success = True
                iDist = iDistMin*(2**a)
                k = (2**(float(b)/float(stackSpace)))
                sigma = (iDist/iDistMin)*sigmaMin*k                
                candidatePoint[4] = round(iDist*(xCap[0,0]+i)) #i increment
                candidatePoint[5] = round(iDist*(xCap[0,1]+j)) #j increment
                DxCap = diffOfGauss[a,b][i,j] - (gMatrix.dot(HMatrix.dot(gMatrix.transpose()))/2)
                candidatePoint[6] = sigma 
                break  
            else:
                #round these values ---
                candidatePoint[2] += round(xCap[0,0])
                candidatePoint[3] += round(xCap[0,1])  
              
            tries += 1
            
        if (success):
            if (abs(DxCap) >= 0.8*CTreshold):
                if (Harris < ((curveRatio+1)**2)/curveRatio):
                    ipolKPoints = np.insert(ipolKPoints, 0, candidatePoint)
#                 else:
#                     print "point discarded based on Harris"
#             else:
#                 print "point discarded based on threshold"
#         else:
#             print "point discarded because unstable"        
                    
    #delete the last record
    for i in range(0,8):
        ipolKPoints = np.delete(ipolKPoints, ipolKPoints.shape[0]-1) 
 
    #reshape the array into a matrix
    ipolKPoints = np.reshape(ipolKPoints,(ipolKPoints.shape[0]/8,8))

def Grad_Orient():
    #pre-compute gradient and orientation information
    for a in range(numOctaves):
        tempOrient = np.empty([oStack[a, 2].shape[0], oStack[a, 1].shape[1]])
        tempGrad = np.empty([oStack[a, 2].shape[0], oStack[a, 1].shape[1]])
        for i in range(1, oStack[a, 2].shape[0] - 1):
            for j in range(1, oStack[a, 2].shape[1] - 1):
                tempGrad[i, j] = np.sqrt((oStack[a, 2][i + 1, j] - oStack[a, 2][i - 1, j]) ** 2 + (oStack[a, 2][i, j + 1] - oStack[a, 2][i, j - 1]) ** 2)
                try:
                    tempOrient[i, j] = np.arctan((oStack[a, 2][i, j + 1] - oStack[a, 2][i, j - 1]) / (oStack[a, 2][i + 1, j] - oStack[a, 2][i - 1, j]))
                except: 
                    print"Exception: division by zero"
        mStack[a] = tempGrad
        orStack[a] = tempOrient
                            
def Orientation():
    #determine the orientation of feature points
    global ipolKPoints
    global finKPoints

    for a in range(0,ipolKPoints.shape[0]):
        oBin = np.zeros([numBins]) #initialized
        sigma = ipolKPoints[a,6]
        oc = ipolKPoints[a,0]
        iDist = iDistMin*(2**oc)
        y = ipolKPoints[a,2]
        x = ipolKPoints[a,3] #may have to change to x and y
        W = np.abs(((x - (3*lbda*sigma))/iDist) - ((x + (3*lbda*sigma))/iDist))
        H = np.abs(((y - (3*lbda*sigma))/iDist) - ((y + (3*lbda*sigma))/iDist))
        if (0 < x-W and x+W < orStack[oc].shape[1] and 0 < y-H and y+H < orStack[oc].shape[0]):
            mPatch = np.empty([H,W])
            orPatch = np.empty([H,W])
            for i in range(0,int(H)):
                for j in range(0,int(W)):
                    #fill the patches
                    m = int(y-(H/2))+i; n = int(x-(H/2))+j
                    mPatch[i,j] = mStack[oc][m,n]
                    orPatch[i,j] = orStack[oc][m,n]
                    
            #gaussian blur the mPatch
            gPatch = cv2.GaussianBlur(mPatch,(0,0),lbda*sigma)
            for i in range(0,int(H)):
                for j in range(0,int(W)):
                    binNum = np.round(np.mod((numBins/(2*np.pi))*(orPatch[i,j]),2*np.pi))
                    oBin[binNum] += gPatch[i,j]
        
            #get the orientation of the point and add to the vector
            maxOrient = np.max(oBin)
            ipolKPoints[a,7] = maxOrient
            finKPoints = np.insert(finKPoints, 0, ipolKPoints[a])
        #else:
            #print "Point too close to border"
    for i in range(0,8):
        finKPoints = np.delete(finKPoints, finKPoints.shape[0]-1) 
    finKPoints = np.reshape(finKPoints,(finKPoints.shape[0]/8,8))

def convertKeyPoints(keyPoints):
    
    convKeyPoints = np.empty([1],dtype=object)
    
    for i in range(keyPoints.shape[0]):
        y = keyPoints[i,2]
        x = keyPoints[i,3]
        sigma = keyPoints[i,6]
        size = int(2*lbdaDesc*sigma)
        theta = keyPoints[i,7]
        oc = int(keyPoints[i,0])
        kp = cv2.KeyPoint(x,y,_size = size, _angle = theta*180/np.pi,_octave = oc)
        convKeyPoints = np.insert(convKeyPoints,convKeyPoints.shape[0],kp,axis=0)       
    convKeyPoints = np.delete(convKeyPoints,0) 
    return convKeyPoints
 

def SiftPoints(imageURL,displayImage):
    
    image, origImage = getImage(imageURL)
    if not(image is None):
        BuildOctaves(image)
        DiffOfGauss()    
        LocalExtrema()    
        KeyPointsFilter()   
        Grad_Orient()    
        Orientation()
        if (displayImage):
            cPoints = convertKeyPoints(finKPoints)
            outImage = cv2.drawKeypoints(origImage,cPoints,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            #outImage = cv2.drawMatches(origImage,cPoints,None,None,None,-1,-1)
            return finKPoints, outImage
        return finKPoints, None
    else:
        return None, None
   
# def showMatches(image1, image2): 
