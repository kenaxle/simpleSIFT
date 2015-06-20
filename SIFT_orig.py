'''
Created on Apr 1, 2015

@author: kena
'''
####################################################################
#
# Python module to perform SIFT feature detection. this is
# a simple implementation of the SIFT algorithm (Lowe 2004) 
#
####################################################################

import cv2
import numpy
from numpy import dtype, int64, ndarray
from Carbon.Events import alphaLock
from cv2 import INTER_LINEAR


# Variables******

#sigmaMin =  
sigmaMin = 0.50 #0.5
sigma = sigmaMin
stackSpace = 3
numOctaves = 3
CDoG = 0.03 # standard for a scalespace of 3
curveRatio = 10

k = numpy.sqrt(2)
dWindow = 16
numBins = 8
numHist = 4

ExtremaThreshold = 0.03 #this is given in the paper however it should be a value based on the number of scales per octave

#k = 2^(1/stackSpace)

xCap = numpy.empty([1,2])
dubya = numpy.empty([1])

#structure for holding the octaves of stacks
oStack = numpy.empty([numOctaves,stackSpace+1], dtype=object)
#structure for holding the difference of Gaussians
diffOfGauss = numpy.empty([numOctaves,stackSpace], dtype=object) 
#structure for holding the keypoints at different scales
kPoints = numpy.empty([1,4],dtype=numpy.int)
numpy.delete(kPoints,0)
ipolKPoints = numpy.empty([1,8],dtype = numpy.int)


mStack = numpy.empty([numOctaves], dtype = object)
orStack = numpy.empty([numOctaves], dtype = object)


# read the image (grayscale) into a varible. this is already stored as a multidim array
imageRead = cv2.imread('lena256.jpg',cv2.CV_LOAD_IMAGE_GRAYSCALE)
image = cv2.resize(imageRead,(int(imageRead.shape[0]/0.5),int(imageRead.shape[1]/0.5)),0,0,INTER_LINEAR)
initImage = cv2.normalize(image.astype('float'), None, 0.00, 1.00, cv2.NORM_MINMAX)

    

#builds the convolution kernel based on sigma supplied
def BuildKernel(sigma):
    gKernel = numpy.zeros([5,5])
    for x in range(-2,3):
        for y in range(-2,3):
            gKernel[x+2,y+2] = (1/(2*numpy.pi*((k*sigma)**2)))*(numpy.e**(-((x**2)+(y**2))/(2*((k*sigma)**2))))
    return gKernel

#performs gaussian blur transform on the image using the pre-calculated kernel

def GBlur(image, kernel):
    outputArr = cv2.filter2D(image,-1,kernel) #perform convolution using function of openCV library
    return outputArr       


#initialize the first stack position
   
def BuildOctaves():
    
    
    #compute the first octave. The initial image is resized and Interpolated by a factor of 2
#     initImage = cv2.resize(image,(int(image.shape[0]/0.5),int(image.shape[1]/0.5)),0,0,INTER_LINEAR)
    
    oStack[0,0] = initImage#GBlur(initImage,kernel)
    
    for j in range(1,stackSpace+1):
        sigma = 1.6*(k**j)
        kernel = BuildKernel(sigma)
        oStack[0,j] = GBlur(oStack[0,0], kernel)
    
    
    #build other octaves
    for i in range(1,numOctaves):
        #sigma = sigmaMin*(2**(i-1)) #reconfigure sigma based on the octave
        #kernel = BuildKernel(sigma)
        
        oStack[i,0] = cv2.resize(oStack[i-1,0],(oStack[i-1,0].shape[0]/2,oStack[i-1,0].shape[1]/2),0,0)        
        #oStack[i,0] = cv2.resize(oStack[i-1,0],(oStack[i-1,0].shape[0]/(0.5)),(oStack[i-1,0].shape[1]/(0.5)),0,0)
        #print `oStack[i-1,0].shape`+" --> "+`oStack[i,0].shape`
        
        for j in range(1,stackSpace+1):
            sigma = 1.6*k**(j+(2*i))
            kernel = BuildKernel(sigma)
            oStack[i,j] = GBlur(oStack[i,0], kernel)
                         
def DiffOfGauss():
    for i in range(numOctaves):
        for j in range(stackSpace):
            diffOfGauss[i,j] = oStack[i,j] - oStack[i,j+1]

def LocalExtrema():
    global kPoints
    for a in range(numOctaves):
        #kPoints = numpy.zeros([diffOfGauss[a,0].shape[0],diffOfGauss[a,0].shape[1]]) #structure to hold the keypoints
        for b in range(1,stackSpace-1):
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
                         
# The original image was not gaussian blurred so comparisson cannot be made for finding Maxima                    
#                          (diffOfGauss[a,b][i,j] > diffOfGauss[a,b-1][i-1,j-1]) and 
#                          (diffOfGauss[a,b][i,j] > diffOfGauss[a,b-1][i-1,j]) and 
#                          (diffOfGauss[a,b][i,j] > diffOfGauss[a,b-1][i-1,j+1]) and
#                      
#                          (diffOfGauss[a,b][i,j] > diffOfGauss[a,b-1][i+1,j-1]) and 
#                          (diffOfGauss[a,b][i,j] > diffOfGauss[a,b-1][i+1,j]) and 
#                          (diffOfGauss[a,b][i,j] > diffOfGauss[a,b-1][i+1,j+1]) and
#                      
#                          (diffOfGauss[a,b][i,j] > diffOfGauss[a,b-1][i,j-1]) and
#                          (diffOfGauss[a,b][i,j] > diffOfGauss[a,b-1][i,j]) and
#                          (diffOfGauss[a,b][i,j] > diffOfGauss[a,b-1][i,j+1]) and
                    
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b+1][i-1,j-1]) and 
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b+1][i-1,j]) and 
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b+1][i-1,j+1]) and
                     
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b+1][i+1,j-1]) and 
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b+1][i+1,j]) and 
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b+1][i+1,j+1]) and
                     
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b+1][i,j-1]) and
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b+1][i,j]) and
                         (diffOfGauss[a,b][i,j] > diffOfGauss[a,b+1][i,j+1]))): #the point is a max or a min
                    
                        kPoints = numpy.insert(kPoints, 0, [a,b,i,j]) #(kPoints,1,[a,b,i,j],axis=0)
                        
    #delete the last record. this is needed to remove the initial values in the array
    for i in range(0,4):
        kPoints = numpy.delete(kPoints, kPoints.shape[0]-1)   
    
#     for i in range(0,kPoints.shape[0],4):
#         print "Point: [ " +`kPoints[i]`+", "+`kPoints[i+1]`+", "+`kPoints[i+2]`+", "+`kPoints[i+3]`+" ]" 
                
def KeyPointsFilter():
    tempkPoints = numpy.empty([1,4],dtype = numpy.int)  
    global ipolKPoints 
    
    print "initial points "+`kPoints.shape[0]/4`  
    
    #validpoint(a,b,i,j,x,y,sigma,dubya)
    #Elliminate all below the threshold
    for i in range(0,kPoints.shape[0],4):
        if (diffOfGauss[kPoints[i],kPoints[i+1]][kPoints[i+2],kPoints[i+3]] >= 0):
            tempkPoints = numpy.insert(tempkPoints,0,[kPoints[i],kPoints[i+1],kPoints[i+2],kPoints[i+3]])
    #delete the empty last value
    for i in range(0,4):
        tempkPoints = numpy.delete(tempkPoints, tempkPoints.shape[0]-1) 
 
 
    #interpolate the points
    for inc in range(0,tempkPoints.shape[0],4):
        # 8 dimensional array
        candidatePoint = numpy.array([tempkPoints[inc],tempkPoints[inc+1],tempkPoints[inc+2],tempkPoints[inc+3],0,0,0,0],dtype = float)
        success = False 
        tries = 0
        patchSize = int((1/(2**candidatePoint[0]))*1.6*20)
        #also get rid of points that are too close to the image bounds
        while (tries < 5 and 
               ((0 < candidatePoint[2] - (2*patchSize) and candidatePoint[2] + (2*patchSize) < diffOfGauss[candidatePoint[0],1].shape[0]-1) and 
                (0 < candidatePoint[3] - (2*patchSize) and candidatePoint[3] + (2*patchSize) < diffOfGauss[candidatePoint[0],1].shape[1]-1))):
            a = candidatePoint[0]; b = candidatePoint[1]
            i = candidatePoint[2]; j = candidatePoint[3]

#             print "Try #: "+`tries`+" with Candidate point:  [ " +`a`+", "+`b`+", "+`i`+", "+`j`+" ]"
            #scale difference is not needed because all points are found on the first scale
            
#             gMatrix = numpy.matrix([((diffOfGauss[a,b+1][i,j])-(diffOfGauss[a,b-1][i,j]))/2, #scale diff
#                                     ((diffOfGauss[a,b][i+1,j])-(diffOfGauss[a,b][i-1,j]))/2, #column difference
#                                     ((diffOfGauss[a,b][i,j+1])-(diffOfGauss[a,b][i,j-1]))/2]) #row difference
            gMatrix = numpy.matrix([((diffOfGauss[a,b][i+1,j])-(diffOfGauss[a,b][i-1,j]))/2, #column difference
                                    ((diffOfGauss[a,b][i,j+1])-(diffOfGauss[a,b][i,j-1]))/2]) #row difference

            HMatrix = numpy.matrix([[(diffOfGauss[a,b][i+1,j])+(diffOfGauss[a,b][i-1,j])-(2*diffOfGauss[a,b][i,j]),((diffOfGauss[a,b][i+1,j+1])-(diffOfGauss[a,b][i+1,j-1])-(diffOfGauss[a,b][i-1,j+1])+(diffOfGauss[a,b][i-1,j-1]))/4],
                                      [((diffOfGauss[a,b][i+1,j+1])-(diffOfGauss[a,b][i+1,j-1])-(diffOfGauss[a,b][i-1,j+1])+(diffOfGauss[a,b][i-1,j-1]))/4,(diffOfGauss[a,b][i,j+1])+(diffOfGauss[a,b][i,j-1])-(2*diffOfGauss[a,b][i,j])]])
                
#             HMatrix = numpy.matrix([[(diffOfGauss[a,b+1][i,j])+(diffOfGauss[a,b-1][i,j])-(2*diffOfGauss[a,b][i,j]),((diffOfGauss[a,b+1][i+1,j])-(diffOfGauss[a,b+1][i-1,j])-(diffOfGauss[a,b-1][i+1,j])+(diffOfGauss[a,b-1][i-1,j]))/4,((diffOfGauss[a,b+1][i,j+1])-(diffOfGauss[a,b+1][i,j-1])-(diffOfGauss[a,b-1][i,j+1])+(diffOfGauss[a,b-1][i,j-1]))/4],
#                                     [((diffOfGauss[a,b+1][i+1,j])-(diffOfGauss[a,b+1][i-1,j])-(diffOfGauss[a,b-1][i+1,j])+(diffOfGauss[a,b-1][i-1,j]))/4,(diffOfGauss[a,b][i+1,j])+(diffOfGauss[a,b][i-1,j])-(2*diffOfGauss[a,b][i,j]),((diffOfGauss[a,b][i+1,j+1])-(diffOfGauss[a,b][i+1,j-1])-(diffOfGauss[a,b][i-1,j+1])+(diffOfGauss[a,b][i-1,j-1]))/4],
#                                     [((diffOfGauss[a,b+1][i,j+1])-(diffOfGauss[a,b+1][i,j-1])-(diffOfGauss[a,b-1][i,j+1])+(diffOfGauss[a,b-1][i,j-1]))/4,((diffOfGauss[a,b][i+1,j+1])-(diffOfGauss[a,b][i+1,j-1])-(diffOfGauss[a,b][i-1,j+1])+(diffOfGauss[a,b][i-1,j-1]))/4,(diffOfGauss[a,b][i,j+1])+(diffOfGauss[a,b][i,j-1])-(2*diffOfGauss[a,b][i,j])]])
#                 

            Trace = numpy.trace(HMatrix)
            Det = numpy.linalg.det(HMatrix)
            Harris = (Trace**2)/Det
            
            #xCap = gMatrix.dot(-HMatrix.getI())
            xCap = gMatrix.dot(-(numpy.linalg.inv(HMatrix)))
            dubya = diffOfGauss[a,b][i,j]+0.5*xCap.dot(gMatrix.transpose())
            
            #***********
            #this may have to be recalculated
            interDist = (0.5*(candidatePoint[0]+1)*2)      
                            
            #absolute coords
            candidatePoint[4] = round(interDist*(xCap[0,0]+candidatePoint[2])) #i increment
            candidatePoint[5] = round(interDist*(xCap[0,1]+candidatePoint[3])) #j increment
            candidatePoint[6] = dubya[0,0]   
            
            #round these values ---
            candidatePoint[2] += round(xCap[0,0])
            candidatePoint[3] += round(xCap[0,1])  
              
            if (abs(xCap[0,0]) < 0.6 and abs(xCap[0,1]) < 0.6 ): #only test for difference of x,y
                success = True
                break
            #increment number of tries    
            tries += 1
            
        if (success):
            #reject low contrast points
#             print "Dubya: "+`candidatePoint[7]`
            if (abs(candidatePoint[6]) >= CDoG):
                if (Harris < ((curveRatio+1)**2)/curveRatio):
                    ipolKPoints = numpy.insert(ipolKPoints, 0, candidatePoint)
#                 else:
#                     print "point discarded based on Harris"
#             else:
#                 print "point discarded based on threshold"
#         else:
#             print "point discarded because unstable"        
                    
    #delete the last record
    for i in range(0,8):
        ipolKPoints = numpy.delete(ipolKPoints, ipolKPoints.shape[0]-1) 
 
    #reshape the array into a matrix
    ipolKPoints = numpy.reshape(ipolKPoints,(ipolKPoints.shape[0]/8,8))
   # print "final points "+`ipolKPoints.shape[0]`  
            
#for testing the images in the stack

#a is the octave
#b is the stack level; for this implementation the stack level should be invarible because only one in stack

def Gradient():
    for a in range(numOctaves):
        tempOrient = numpy.empty([oStack[a, 1].shape[0], oStack[a, 1].shape[1]])
        tempGrad = numpy.empty([oStack[a, 1].shape[0], oStack[a, 1].shape[1]])
        for i in range(1, oStack[a, 1].shape[0] - 1):
            for j in range(1, oStack[a, 1].shape[1] - 1):
                tempGrad[i, j] = numpy.sqrt((oStack[a, 1][i + 1, j] - oStack[a, 1][i - 1, j]) ** 2 + (oStack[a, 1][i, j + 1] - oStack[a, 1][i, j - 1]) ** 2)
                try:
                    tempOrient[i, j] = numpy.arctan((oStack[a, 1][i, j + 1] - oStack[a, 1][i, j - 1]) / (oStack[a, 1][i + 1, j] - oStack[a, 1][i - 1, j]))
                except: 
                    print"division by zero"
        mStack[a] = tempGrad
        orStack[a] = tempOrient
                            

def Orientation():
    global ipolKPoints
    oBin = numpy.zeros([numBins])
    vect = numpy.empty([16],dtype=object)
    
    for a in range(0,ipolKPoints.shape[0]):
        patchSize = int(((1.6*20)/(2**ipolKPoints[a,0])))
        for i in range(int(ipolKPoints[a,2]-int(patchSize/2)),int(ipolKPoints[a,2]+int(patchSize/2))):
                for j in range(int(ipolKPoints[a,3]-int(patchSize/2)),int(ipolKPoints[a,3]+int(patchSize/2))):
                    binNum = numpy.round((numBins*(orStack[ipolKPoints[a,0]][i,j]+numpy.pi))/(2*numpy.pi))
                    temp = oBin[binNum]
                    temp += mStack[ipolKPoints[a,0]][i,j] #add the magnitude to the bin
                    oBin[binNum] = temp
        
        ipolKPoints[a,7] = oBin.argmax(0)
        print "Point: [ " +`ipolKPoints[a]`+", "+`ipolKPoints[a+1]`+", "+`ipolKPoints[a+2]`+", "+`ipolKPoints[a+3]`+" Orientation: "+`ipolKPoints[a+7]`+" ] #: "+`a` 


#the gradients and magnitudes should be gaussian weighted. 
#my implementation does not at this time

def Descriptor(a,x,y):
    oBin = numpy.zeros([numBins])
    vect = numpy.empty([16],dtype=object)
    cell = 0
    patchSize = int((1/(2**a))*1.6*20)
    for di in range(int(x-(patchSize*numHist/2)),int(x+(patchSize*numHist/2)),patchSize):
        for dj in range(int(y-(patchSize*numHist/2)),int(y+(patchSize*numHist/2)),patchSize):
    
            for i in range(di,di+patchSize):
                for j in range(dj,dj+patchSize):
                    binNum = numpy.round((numBins*(orStack[a][i,j]+numpy.pi))/(2*numpy.pi))
                    temp = oBin[binNum]
                    temp += mStack[a][i,j] #add the magnitude to the bin
                    oBin[binNum] = temp
                    
            vect[cell] = oBin
            cell += 1
    
    print vect
                    #print " Bin #: "+`binNum`
                    #print"Orientation: "+`orStack[a][i,j]`
#                print "gradient: "+`mStack[a][i,j]`
    #print oBin       


def outPutPoints():
    for i in range(0,ipolKPoints.shape[0],8):
        patch = int(((1/(2**ipolKPoints[i]))*1.6*20)/2)
        for s in range(ipolKPoints[i+4]-(patch),ipolKPoints[i+4]+patch):
            for t in range(ipolKPoints[i+5]-(patch),ipolKPoints[i+5]+patch):
#                 image[ipolKPoints[i+4],ipolKPoints[i+5]] = 255
                image[s,t] = 255
    cv2.imwrite('Final lena.jpg',image) 
       

def PrintStack(oct):
    for i in range(0,stackSpace+1):
        cv2.imwrite('Stack' + `oct` +'lena'+`i`+'.jpg',oStack[oct,i])
        
def PrintDoG(oct):
    for i in range(0,stackSpace+2):
        cv2.imwrite('Diff' + `oct` +'lena'+`i`+'.jpg',diffOfGauss[oct,i])        

#print cv2.resize(image,(image.shape[0]/2,image.shape[1]/2))
BuildKernel(sigma)

BuildOctaves()

#print oStack

DiffOfGauss()

#print diffOfGauss
 
LocalExtrema()

# 
KeyPointsFilter()

Gradient()

Orientation()

outPutPoints()

i = 0
Descriptor(ipolKPoints[i],ipolKPoints[i+2],ipolKPoints[i+3])


# for i in range(0,4*200,4):
#     print `i`+": [ " +`kPoints[i]`+", "+`kPoints[i+1]`+", "+`kPoints[i+2]`+", "+`kPoints[i+3]`+" ]"

# i = 0
# quadraticInterpolation(kPoints[i],kPoints[i+1], kPoints[i+2], kPoints[i+3])


# for i in range(0,kPoints.shape[0],4):
# KeyPointsValidation()
#print kPoints
#outputArray = GBlur(image, gKernel)
#print oStack
# print outputArray
#cv2.imwrite('output0.jpg',diffOfGauss[0,0])
# cv2.imwrite('output1.jpg',kPointsPerOctave[1])
# cv2.imwrite('output2.jpg',kPointsPerOctave[2])
#imgArray = numpy.array(image)

    
    
