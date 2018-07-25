from path_utilities import get_immediate_subdirectories
from path_utilities import get_last_dir_from_path
from path_utilities import remove_extension 
from path_utilities import str_decode
 
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec 
from gensim.test.utils import get_tmpfile
from collections import namedtuple
from smart_open import smart_open
from tqdm import tqdm

import numpy as np

import requests
import os.path
import tarfile
import codecs
import gensim
import locale
import glob
import sys
import re

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

def parse_args():

    print("Parsing arguments. ")
    # ARGUMENTS    
    # source directory and output directory
    dataset_path = sys.argv[1]          # directory of dataset
    num_clusters = 0                    # number of clusters to generate
    fill_threshold = 1                  # ignore rows filled less
    overwrite = "0"                     # overwrite the distance matrix
    overwrite_plot = "0"                # overwrite plot cache 
    # overwrite is a string, should be "0" for don't overwrite, and "1"
    # for do
    arg_list = [
                dataset_path, 
                num_clusters, 
                overwrite, 
                overwrite_plot, 
                fill_threshold,
               ]
    print("Arguments parsed. ")
    return arg_list

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

def check_valid_file(some_file):
    if not os.path.isfile(some_file):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("")
        print("DIES IST KEIN GÜLTIGER SPEICHERORT FÜR DATEIEN!!!!")
        print("")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit()

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

# RETURNS: a list of fnames and a list of the contents of the files 
def get_document_contents(directory, dataset_path):
    
    # setup output path
    grandparent = Path(Path(os.getcwd()).parent).parent
    dataset_name = path_utilities.get_last_dir_from_path(dataset_path)
    write_path = os.path.join(grandparent, 
                              "cluster-datalake-outputs/", 
                              dataset_name + "--output")    
    
    if not os.path.isdir(write_path):
        os.mkdir(write_path)
    
    # load in the extension index from the output folder
    ext_dict_file_loc = os.path.join(write_path, "extension_index_" 
                                     + dataset_name + ".npy") 
    ext_paths = np.load(ext_dict_file_loc).item()
    
    # "filenames" list of the paths of files
    # "data" list of the contents of files 
    filenames = []
    data = []
    i = 1
    
    # get contents of txt files still in original dataset
    txt_paths = ext_paths.get("txt")
    print("Getting .txt contents from " + dataset_path)
    for path in tqdm(txt_paths):
        if os.path.isfile(path):
            i = i + 1
                
            # add the path of the file to "filenames" 
            filenames.append(path)

            # read the contents of the file and remove newlines
            fread = open(path, "r", errors='backslashreplace')
            contents = fread.read()
            fread.close()
            contents = contents.replace("\n","")
            
            # add the string of the contents of the file to "data"
            data.append(contents)
    
    # get contents of converted files in the other directory
    conv_folders = get_immediate_subdirectories(directory)
    
    # for each folder in the directory (e.g. pdf/ doc/)
    for folder in conv_folders:
        filetype = get_last_dir_from_path(folder)
        if filetype in ["pdf", "doc", "docx"]: #, "xml", "html"]:
            print("Getting ."+folder+" contents")
            for filename in tqdm(os.listdir(folder)):
                path = os.path.join(folder,filename)
                if os.path.isfile(path):
                    i = i + 1
                    
                    # add the non-converted filename to "filenames" 
                    new_name = str_decode(remove_extension(filename))
                    filenames.append(new_name)

                    # read the contents of the file and remove newlines
                    fread = open(path, "r", errors='backslashreplace')
                    contents = fread.read()
                    fread.close()
                    contents = contents.replace("\n","")
                    
                    # add the string of the file contents to "data"
                    data.append(contents)
    
    print("Num total files: ", i)
    print("All directory contents retrieved")
    return filenames, data

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

def doc2vec(dataset_path, dataset_name, write_path, txt_path_list):

    # Convert text to lower-case and strip punctuation/symbols from words
    def normalize_text(text):
        norm_text = text.lower()
        # Replace breaks with spaces
        norm_text = norm_text.replace('<br />', ' ')
        # Pad punctuation with spaces on both sides
        norm_text = re.sub(r"([\.\",\(\)!\?;:])", " \\1 ", norm_text)
        return norm_text

    alldata_path = os.path.join(write_path, 'alldata-id_' + dataset_name + '.txt') 

    if not os.path.isfile(alldata_path):

        # Collect & normalize test/train data
        print("Cleaning up dataset...")
            
        # list of the absolute paths of every text file
        print(" %i files" % (len(txt_path_list)))
 
        # for each file "txt"
        for i, txt in tqdm(enumerate(txt_path_list)):
            with smart_open(txt, "rb") as t:

                try: 
                    # "one_text" is the whole document
                    one_text = t.read().decode("utf-8")
                    for c in control_chars:
                        one_text = one_text.replace(c, ' ')
                    one_text = normalize_text(one_text)
                    all_lines.append(one_text)
                except UnicodeDecodeError:
                    
                    # we skip this file, but we need to preserve index pos
                    all_lines.append(" ")
                    continue

        # Save to disk for instant re-use on any future runs
        with smart_open(alldata_path, 'wb') as f:
            for idx, line in enumerate(all_lines):
                num_line = u"_*{0} {1}\n".format(idx, line)
                f.write(num_line.encode("utf-8"))

    assert os.path.isfile(alldata_path), "alldata unavailable"
    print("Success, alldata is available for next steps.")

    #===================================================================
    #=#BLOCK#=#: Read in alldata 
    #===================================================================

    # this data object class suffices as a `TaggedDocument` 
    # (with `words` and `tags`)
    # plus adds other state helpful for our later evaluation/reporting

    with smart_open(alldata_path, 'rb', encoding='utf-8') as alldata:
        alldata_list = list(alldata)
        print("Iterating up to: ", len(alldata_list))
    with smart_open(alldata_path, 'rb', encoding='utf-8') as alldata:
        documents = [TaggedDocument(doc, [i]) for i, doc in tqdm(enumerate(alldata))]
        model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

        fname = get_tmpfile(os.path.join(write_path, "doc2vec_model_" + dataset_name))
        model.save(fname)
        model = Doc2Vec.load(fname)  # you can continue training with the loaded model!

    return

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

# MAIN PROGRAM:
def runflow(dataset_path): 
   
    #===================================================================
    #=#BLOCK#=#: Get read and write paths for cluster functions 
    #===================================================================
    print("Getting read and write paths for cluster functions. ")  
  
    # check if the dataset location is a valid directory 
    check_valid_dir(dataset_path)
   
    # get its absolute path
    dataset_path = os.path.abspath(dataset_path)
    
    # the name of the top-level directory of the dataset
    dataset_name = get_last_dir_from_path(dataset_path)
     
    # get converted file location and output location
    out_dir = os.path.join(dataset_path, 
                           "../" + "converted-" + dataset_name)
    
    # define the write path for the entire program
    write_path = "../../cluster-datalake-outputs/" + dataset_name + "--output/"
    if not os.path.isdir(write_path):
        os.system("mkdir " + write_path)
    print("All results printing to " + write_path)
    
    # get absolute paths 
    out_dir = os.path.abspath(out_dir)
    write_path = os.path.abspath(write_path)
    
    # get the location of the extension index file
    print("Finding extension index file. ")
    ext_dict_file_loc = os.path.join(write_path, "extension_index_"
                                     + dataset_name + ".npy")
    
    # check if the above paths are valid 
    check_valid_dir(out_dir)
    check_valid_file(ext_dict_file_loc)
    
    # load the extension to path dict
    ext_to_paths_dict = np.load(ext_dict_file_loc).item()
    csv_path_list = []
    txt_path_list = []
    if "csv" in ext_to_paths_dict:
        csv_path_list = ext_to_paths_dict["csv"]
    if "txt" in ext_to_paths_dict:
        txt_path_list = ext_to_paths_dict["txt"]
 
    # location of files converted to csv format
    csv_dir = os.path.join(out_dir, "csv/")
    converted_csv_list = os.listdir(csv_dir)

    doc2vec(dataset_path, dataset_name, write_path, csv_path_list)

    return

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

def main():
    
    arg_list = parse_args()
    dataset_path = arg_list[0]
    num_clusters = arg_list[1]
    overwrite = arg_list[2]
    overwrite_plot = arg_list[3]
    fill_threshold = arg_list[4]

    print("Don't run this file standalone. ")
    runflow(dataset_path) 
    return

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 


