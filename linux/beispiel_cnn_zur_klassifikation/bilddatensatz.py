import os
from os import listdir
from os.path import isdir, isfile, join
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

  
class bilddatensatz:
    
    #
    # Geht alle Unterverzeichniss des angegebenen
    # Wurzelverzeichnisses durch und
    # generiert eine Liste der Form:
    #
    # [ ["data/bikes/jfksdj43.jpg", "bikes",
    #   ["data/cars/bvcnm401.jpg", "cars"],
    #   ...
    # ]
    #
    def __init__(self, root_folder, img_size, inputs_fuer_VGG16=False):
        
        print("!!!")
        
        self.img_size = img_size
        
        self.inputs_fuer_VGG16 = inputs_fuer_VGG16
        
        self.all_training_items = []
       
        self.class_names = \
            [d for d in listdir(root_folder)
             if isdir(os.path.join(root_folder,d))]

        print("Unter dem Verzeichnis\n\t", root_folder,
              "\nhabe ich folgende Unterordner/Klassen gefunden:")
        print(self.class_names)
        
        self.nr_classes = len(self.class_names)
        
        # For each subfolder ...
        for class_id, class_name in enumerate(self.class_names):
            
            subfolder_name = root_folder + "/" + class_name + "/"
            
            filenames = \
                [subfolder_name + f
                 for f in listdir(subfolder_name)
                 if isfile(join(subfolder_name, f))]
            
            print("{} Dateien im Unterverzeicnis {}".format(len(filenames),
                                                     subfolder_name) )
            
            # For each image filename in current subfolder ...
            for filename in filenames:
                
                teacher_vec = np.zeros( self.nr_classes )
                teacher_vec[class_id] = 1.0
                
                self.all_training_items.append(
                    [filename,
                     class_id,
                     class_name,
                     teacher_vec] )
        
        self.nr_images = len(self.all_training_items)
        print("Insgesamt sind {} Bilder verfügbar".format(self.nr_images))
        
        print("Hier die ersten 3 Einträge:")
        print(self.all_training_items[:3])
        
    
    
    def lade_bild(self, absolute_filename):
        """
        Lade ein Bild aus der angegebenen Datei
        und mache es (Farbkanal-spezifisch) Mittelwertfrei
        """
        
        img = cv2.imread(absolute_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                
        img = cv2.resize(img,
                         self.img_size,
                         interpolation=cv2.INTER_AREA)
        
        if self.inputs_fuer_VGG16:                    
            x = img.astype(float)
            x = np.expand_dims(x, axis=0)
            #print("x has shape", x.shape)                
            #print("x has mean", np.mean(x))        
            # From the VGG paper:
            # "The only pre-processing we do is subtracting the mean RGB value,
            # computed on the training set, from each pixel."
            #
            # see imagenet_utils.py
            #
            x = preprocess_input(x)
            #print("x has mean", np.mean(x))   
            img_preprocessed = x.reshape((224,224,3))
        else:            
            img_preprocessed = img * (1.0 / 255.0)
        
        return img, img_preprocessed
        
        
       
    def hole_bild_per_index(self, idx):
        """
        Gebe das Bild aus dem Datensatz
        mit dem Index idx zurück.
        """
        
        image_filename  = self.all_training_items[idx][0]
        class_id        = self.all_training_items[idx][1]
        class_name      = self.all_training_items[idx][2]
        teacher_vec     = self.all_training_items[idx][3]
        
        img, img_preprocessed = self.lade_bild(image_filename)
        
        return img, img_preprocessed, \
               class_id, class_name, teacher_vec
    
    
    def hole_irgendein_bild(self):        
        """
        Gebe ein zufälliges Bild zurück
        """
        
        rnd_idx = np.random.randint(0, self.nr_images)
        return self.hole_bild_per_index( rnd_idx )