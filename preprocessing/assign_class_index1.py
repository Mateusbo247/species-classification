#!/usr/bin/env python3

import sys
import argparse
import csv
import os

"""
   Given a CSV file containing rows of the form:

      id_register,id_animal,class_index

   and a bunch of JPEG filenames as

      id_register-image_number.jpg,

   this will assign the class index to each of these filenames in the form

      id_register-class_index-image_number.jpg


   Usage:

      assign_class_index.py -c id_registro-id_animal-id_tipo.csv images/*
"""

# id_registro, id_animal_registro, id_tipo

def main(arguments):

   registro_animal_tipo = csv.reader( args.csv )

   for i in args.files:
      dirname, basename = os.path.split( i )
      id = basename.split('-')[0] # <id_register>-<image_number>.jpg
      image_num = basename.split('-')[1]
      matches = []
      args.csv.seek(0)
      for j in registro_animal_tipo:
         if j[0] == id:
            matches.append( j )
      if len( matches ) == 0:
         print( "> Error: match not found for {}".format( id ) )
      else:
         for counter, m in enumerate( matches, start = 1 ): # The same id can have multiple animals, so there might be different class indexes
            command = "cp" if counter < len( matches ) else "mv"
            print( "{} {} {}-{}-{}".format( command, os.path.join(dirname, basename), os.path.join( dirname, id ), m[2], image_num ) ) # m[2] is the class index
            


if __name__ == '__main__':

   parser = argparse.ArgumentParser()

   parser.add_argument("-c", "--csv", type=argparse.FileType('r'), required=True, help="CSV file of the form 'id_register,id_animal,class_index'")
   parser.add_argument("files", type=str, nargs='+', help="JPEG filenames")

   args = parser.parse_args()

   sys.exit(main(sys.argv[1:]))
