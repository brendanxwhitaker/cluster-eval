from bs4 import BeautifulSoup
from itertools import repeat
from tqdm import tqdm

import pandas as pd
import numpy as np

import path_utilities
import textract
import DFS
import sys
import csv
import os
import re

#=========1=========2=========3=========4=========5=========6=========7=

def parse_args():

    # ARGUMENTS
    dataset_path = sys.argv[1]         # the dataset location
    num_top_exts = int(sys.argv[2])      # use k most frequent extensions 

    return dataset_path, num_top_exts

def check_valid_dir(some_dir):
    if not os.path.isdir(some_dir):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("")
        print("DIES IST EIN UNGÜLTIGES VERZEICHNIS!!!!")
        print("")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit()

#=========1=========2=========3=========4=========5=========6=========7=

''' DOES: converts all the pdfs whose paths are specified in "pdf_paths"
          and puts the resultant text files in "dest/pdf/" '''
def convert_pdfs(pdf_paths, dest):
    num_pdfs = len(pdf_paths)
    print(num_pdfs, " pdfs for conversion")
    for path in tqdm(pdf_paths):      
        output_dir = os.path.join(dest, "pdf")
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        transformed_path = path_utilities.str_encode(path)
        if not os.path.isfile(os.path.join(output_dir, transformed_path + ".txt")):
            os.system("ebook-convert " + path + " " + os.path.join(output_dir, transformed_path + ".txt")) 

def convert_doc(doc_paths, dest):
    num_docs = len(doc_paths)
    print(num_docs, " docs for conversion")
    for path in tqdm(doc_paths): 
        try:     
            output_dir = os.path.join(dest, "doc")
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            transformed_path = path_utilities.str_encode(path)
            os.chdir(output_dir)
            if not os.path.isfile(os.path.join(output_dir, transformed_path + ".txt")):
                f = open(transformed_path+".txt", "w")
                contents = textract.process(path).decode("UTF-8").replace("\n", " ")
                f.write(contents)
                f.close()
        except textract.exceptions.ShellError:
            continue
 
def convert_docx(docx_paths, dest):
    num_docxs = len(docx_paths)
    print(num_docxs, " docxs for conversion")
    for path in tqdm(docx_paths):     
        try:
            output_dir = os.path.join(dest, "docx")
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            transformed_path = path_utilities.str_encode(path)
            os.chdir(output_dir)
            if not os.path.isfile(os.path.join(output_dir, transformed_path + ".txt")):
                f = open(transformed_path+".txt", "w")
                contents = textract.process(path)
                f.write(str(contents))
                f.close()
        except textract.exceptions.ShellError:
            continue

def convert_html(html_paths, dest):
    num_htmls = len(html_paths)
    print(num_htmls, " htmls for conversion")
    for path in tqdm(html_paths):     
        with open(path, 'r', errors="backslashreplace") as content_file:
            contents = content_file.read()
            soup = BeautifulSoup(contents, 'html.parser')         
            output_dir = os.path.join(dest, "html")
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            transformed_path = path_utilities.str_encode(path)
            os.chdir(output_dir)
            if not os.path.isfile(os.path.join(output_dir, transformed_path + ".txt")):
                f = open(transformed_path+".txt", "w")
                f.write(soup.get_text())
                f.close()

def convert_xml(xml_paths, dest):
    num_xmls = len(xml_paths)
    print(num_xmls, " xmls for conversion")
    for path in tqdm(xml_paths):     
        with open(path, 'r', errors="backslashreplace") as content_file:
            contents = content_file.read()
            soup = BeautifulSoup(contents, 'xml')         
            output_dir = os.path.join(dest, "xml")
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            transformed_path = path_utilities.str_encode(path)
            os.chdir(output_dir)
            if not os.path.isfile(os.path.join(output_dir, transformed_path + ".txt")):
                f = open(transformed_path+".txt", "w")
                f.write(soup.get_text())
                f.close()

#=========1=========2=========3=========4=========5=========6=========7=

# RETURNS: list of filepaths which are candidates for conversion.
def get_valid_filenames_struct(dir_list):
    print("size of virtual directory: ", len(dir_list))
    list_valid_exts = [".xls", ".xlsx", ".tsv"]
    list_caps_exts = {".XLS":".xls", ".XLSX":".xlsx", ".TSV":".tsv"}
    valid_list = []
    # for each filename in the directory...
    for path in tqdm(dir_list):
        filename = path_utilities.get_fname_from_path(path)
        length = len(filename)
        valid = False
        # we iterate on the characters starting from end of the string
        pos = length - 1
        # dot pos will be position of the first period from the end,
        # i.e. the file extension dot. If it is still "length" at end,
        # we know that there are no dots in the filename. 
        dot_pos = length
        while (pos >= 0):
            if (filename[pos] == "."):
                dot_pos = pos
                break
            pos = pos - 1
        # if there is a dot somewhere in filename, i.e. if there is an
        # extension...
        if (dot_pos < length):
            extension = filename[dot_pos:length]
            if extension in list_valid_exts:
                valid_list.append(path)
                valid = True
            # if the extension is an ALLCAPS version...
            elif extension in list_caps_exts.keys():
                #new_filename = filename[0:dot_pos] 
                # + list_caps_exts[extension]
                # change it to lowercase and add it to valid_list
                #os.system("mv " + os.path.join(dataset_path, filename) 
                # + " " + os.path.join(dataset_path, new_filename))
                valid_list.append(path)
                valid = True
        if (valid == False):
            print(extension)
            print("This filename is invalid: ", filename)
        
    print("There are ", len(valid_list), " candidates for conversion. ")
    return valid_list

#=========1=========2=========3=========4=========5=========6=========7=

# DOES: converts all the files in valid list to csv, and puts the
#       resultant files in out_dir. 
def convert_tabular(valid_list, out_dir):

    for path in tqdm(valid_list):

        # output will look like <encoded_filepath_w/_extension>.csv.<i>
        encoded_filename = path_utilities.str_encode(path)
        out_path = os.path.join(out_dir, encoded_filename)
        if not os.path.isfile(out_path + ".csv.0"):
            print("out_path: ", out_path)
            print("converting")
            os.system("ssconvert " + path + " " 
                      + out_path + ".csv > /dev/null 2>&1 -s")
    return

#=========1=========2=========3=========4=========5=========6=========7=

def convert_tsv(valid_list, out_dir):

    for path in tqdm(valid_list):

        # output will look like <encoded_filepath_w/_extension>.csv.<i>
        encoded_filename = path_utilities.str_encode(path)
        out_path = os.path.join(out_dir, encoded_filename)
        if not os.path.isfile(out_path + ".csv.0"):
            print("out_path: ", out_path)
            print("converting") 
            with open(path,'r') as tsvin, open(out_path, 'w') as csvout:
                try:
                    tsvin = csv.reader(tsvin, delimiter='\t')
                    csvout = csv.writer(csvout)

                    for row in tsvin:
                        count = int(row[4])
                        if count > 0:
                            csvout.writerows(repeat(row[2:4], count))

                except UnicodeDecodeError:
                    continue
                except MemoryError:
                    print("Memory error, skipping this file. ")
                    continue
    return

#=========1=========2=========3=========4=========5=========6=========7=

# MAIN FUNCTION
def convert(dataset_path, num_top_exts):
    
    check_valid_dir(dataset_path)

    # the name of the top-level directory of the dataset
    dataset_name = path_utilities.get_last_dir_from_path(dataset_path)

    # get its absolute path
    dataset_path = os.path.abspath(dataset_path)
    dest = os.path.join(dataset_path, "../converted-" + dataset_name + "/")
    if not os.path.isdir(dest):
        os.system("mkdir " + dest)
    check_valid_dir(dest)

    # get the script output location
    write_path = os.path.join("../../cluster-datalake-outputs/" + dataset_name + "--output/")

    # get its absolute path
    write_path = os.path.abspath(write_path)
    if not os.path.isdir(write_path):
        os.system("mkdir " + write_path)
    check_valid_dir(write_path)

    # create the destination directories for converted files.  
    csv_dest = os.path.join(dest, "csv/")
    print("csv_dest: ", csv_dest)
    if not os.path.isdir(csv_dest):
        os.system("mkdir " + csv_dest)

    # get a dictionary which maps extension names of the form "csv"
    # to lists of the full paths of files with those extensions in the 
    # dataset.
    # CREATES "extension_index_<dataset_name>.npy"
    ext_locations = DFS.extension_indexer(dataset_path, 
                                          num_top_exts, write_path)

    # if we have extensions with the following names, performs 
    # conversion.
     
    if "pdf" in ext_locations:
        pdf_paths = ext_locations.get("pdf")
        convert_pdfs(pdf_paths, dest)
    if "doc" in ext_locations:
        doc_paths = ext_locations.get("doc")
        convert_doc(doc_paths, dest)
    if "docx" in ext_locations:
        docx_paths = ext_locations.get("docx")
        convert_docx(docx_paths, dest)
    if "html" in ext_locations:
        html_paths = ext_locations.get("html")
        convert_html(html_paths, dest)
    if "xml" in ext_locations:
        xml_paths = ext_locations.get("xml")
        convert_xml(xml_paths, dest)
    
    if "xls" in ext_locations:
        xls_paths = ext_locations.get("xls")
        valid_xls = get_valid_filenames_struct(xls_paths)
        convert_tabular(valid_xls, csv_dest)
    if "xlsx" in ext_locations:
        xlsx_paths = ext_locations.get("xlsx")
        valid_xlsx = get_valid_filenames_struct(xlsx_paths)
        convert_tabular(valid_xlsx, csv_dest)
    if "tsv" in ext_locations:
        tsv_paths = ext_locations.get("tsv")
        valid_tsv = get_valid_filenames_struct(tsv_paths)
        convert_tsv(valid_tsv, csv_dest)
    
def main():
    
    # get the arguments
    dataset_path, num_top_exts = parse_args()
    convert(dataset_path, num_top_exts)
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 


