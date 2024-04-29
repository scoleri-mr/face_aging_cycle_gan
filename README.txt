.
├── checkpoints/
│   ├── resnet_train/
│   ├── transfer_horse2zebra/
│   └── unet_train/
├── dataset_old/
├── dataset_young/
├── extract_faces/
│   ├── model_data/
│   ├── data_generator.py
│   ├── face_extractor_mine.py
│   └── face_extractor.py
├── pretrained_models/
│   ├── horse2zebra_GA.pth
│   └── horse2zebra_GB.pth
├── pretrained_resnet_pytorch.py
├── pretrained_resnet_tensorflow.py
├── resnet_utils.py
├── training_utils.py
├── transfer_learning_utils.py
├── utils.py
├── young2old_dataset_creation.ipynb
└── young2old.ipynb

dataset_old:
folder that contains of elderly people. To have a functioning code you need to extract the faces as shown in young2old_dataset_creation.ipynb. The young2old.ipynb console expects a complete dataset_old_faces with all the extracted faces needed for training.

dataset_young:
folder that contains the images of young people. To have a functioning code you need to extract the faces as shown in young2old_dataset_creation.ipynb. The young2old.ipynb console expects a complete dataset_young_faces with all the extracted faces needed for training.

checkpoints:
folder that contains the latest checkpoint for each trained model.

extract_faces:
folder with models and files used to extract the faces. The code in this folder is copied from https://github.com/kb22/Create-Face-Data-from-Images. I edited the original face_extractor.py creating the 'face_extractor_mine.py'. In my version the square around the faces is bigger, it includes a bigger portion of the face and some elements around it. 

pretrained_models:
folder that contains the pre-trained horse2zebra model from the cyclegan paper. It contains both the generators GA and GB in pth.

pretrained_resnet_pytorch.py:
used for the transfer learning task. Contains the class 'my_Resnet_Generator()' in which is defined the ResNet structure used to load the pretrained pytorch models found in pretrained_models.

pretrained_resnet_pytorch.py:
used for the transfer learning task. Contains the class 'my_ResnetGenerator_tf()' in which the same resnet is defined in tensorflow. 

transfer_learning_utils.py: 
contains all the useful functions to convert the model from pytorch to a model usable in tensorflow. In particular, we have the 'get_tensorflow_model' function that takes the weights from the pth file and transfers them to a newly defined resnet in tensorflow. It also contains the 'freeze_layers' function that is used to find the last residual block and freeze everything before that block. This way only the last residual block and everything after will be trained. 

resnet_utils.py:
file that contains all the functions used to create the resnet generator I trained for the second task (cycle-GAN with resnet generator). This resnet will only contain 6 residual blocks, instead of the 9 that are present in the resnets used for transfer learning.

training_utils.py:
contains the function to initialize the loss functions and the train step function used for training.

utils.py:
contains all other useful functions like those used to preprocess the images before training, produce output images witha given model.

young2old_dataset_creation.ipynb:
detailed notebook that explaines how the datasets were collected (including the scraping script used) and how the faces were extracted. Note: part of the dataset was collected with this script last year, the script may capture very different images if run now. 

young2old.ipynb:
core notebook that contains the most important part of the project. It's divided in different sections that are all very detailed: preprocessing, training (three models), visual testing, quantitative evaluation.

SOURCES:
The face extraction uses the already mentioned github folder https://github.com/kb22/Create-Face-Data-from-Images. 

The code in 'young2old' in the section "train the cycle-GAN using the pix2pix generators and discriminators' is strongly inspired by the code in the tensorflow tutorial about the cycle-GAN: https://www.tensorflow.org/tutorials/generative/cyclegan.









