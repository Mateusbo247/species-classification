This file guides the use of the code assign_class_index1.py
To run the code, use the following command:

pyhton3 assign_class_index.py -c id_registro-id_animal-id_tipo.csv images/*

Where:
- id_registro-id_animal-id_tipo.csv is a file that contains the identification of each record, with its class.
- images is the folder that contains the images that should be renamed.

The code will rename each image according to its record and class, in the following format:

id_register-class_index-image_number.jpg

id: SissGeo record identification;
class: Class present in the image;
image number: image number for records that have more than one image.
