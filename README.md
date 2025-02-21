# Trabajo PIMD
Deeplearning proyect:
## Descripción del proytecto:
Experimentar en la creación de redes neuronales convolucionales 
para taréas de clasificación

### Librerías utilizadas:
- Pytorch
- Pillow
- Matplotlib
- Albumentations
- OpenCV
- Numpy
- Tqdm
- OS
- Json

## Descripción de los códigos:
### Primeras pruebas con CNNs para la detección de estrés:
PIMDTrain.py : red nueronal creada y entrenada desde cero

### Pruebas con modelos preentrenados con el dataset ImageNet y arquitectura ResNet para la detección de estrés:
- PIMDResNet.py
- 7.PIMDTest.py : testing de la red neruonal con validación gráfica
- PIMDCamara.py : código interactivo para ver el funcionamiento a tiempo real

### PIMDPrueba2 para la detección de rasgos físicos:
- Data/Razas : Dataset con las imágenes
- Pesos/LossTrainPIMD2.json : resultados del entrenamiento
- PIMD2.pth : pesos de la red entrenada
- PIMD2ResNet.py : Entrenamiento de la red
- PIMD2Camara.py : código interactivo para ver el funcionamiento a tiempo real
