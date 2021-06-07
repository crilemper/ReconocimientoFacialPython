import cv2
import os   
import numpy as np

dataPath = 'C:/Users/Cristopher/Desktop/Reconocimiento/Data'
peopleList = os.listdir(dataPath)
print('Lista de persona: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo las imagenes')
    
    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName,0))
        image = cv2.imread(personPath+'/'+fileName,0)
    
    label = label + 1

face_recongnizer = cv2.face.EigenFaceRecognizer_create()

#Entrenando el recnocedor de rostros
print('Entrenando...')
face_recongnizer.train(facesData, np.array(labels))

#Almacenando el modelo obtenido
face_recongnizer.write('modeloEigenFace.xml')
print('Modelo almacenado...')
