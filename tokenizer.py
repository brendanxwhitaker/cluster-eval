import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import path_utilities
import pylab as pl
import numpy as np
import DFS
import sys
import os

''' PARAMETER: a list of all filenames with extensions
    RETURNS: a list of filenames without the extensions '''
def remove_all_extensions(filenames):
    filenames_no_extensions = []
    extensions = []
    for filename in filenames:
        extension = ""
        length = len(filename)
        for ch in filename[::-1]:
            if ch==".":
                break
            # "extension" contains just the extension from a filename
            extension = ch + extension
            length = length - 1 
        filenames_no_extensions.append(filename[0:length-1])
        extensions.append(extension.lower())
        #print(filename)
        #print(extension)
    
    return filenames_no_extensions, extensions

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: a single filename
    RETURNS: a list of all tokens with "_" "-" and "._" removed '''
def generate_tokens(filename):
    # replace dashes with underscores, and .dashes with underscores
    # in each filename
    if filename[:2]!="._":
        filename = filename.replace("-","_")
        # create a list of tokens, splitting by underscores
        old_tokens =  filename.split("_")
        new_tokens = []
        # removes all empty tokens
        for x in old_tokens:
            if x!="":
                new_tokens.append(x)
    return new_tokens
        
#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: a list of filenames
    DOES: sorts a dictionary with the counts of each token
    RETURNS: a list of sorted tokens and a list of sorted counts '''
def count_and_sort_tokens(filenames, write_path):
    # a dict mapping tokens to the count of how many times they appear
    token_count_dict = {}
    # for each filename
    for fn in filenames:
        #split the filename into tokens using delimiters like "_"
        tokens = set(generate_tokens(fn))
        # for each token
        for token in tokens:
            # if the token is already in our dict
            if token in token_count_dict.keys():
                # grab the old count
                old_count = token_count_dict.get(token)
                # increment and update the count in the dict              
                token_count_dict.update({token:old_count+1})
            else:
                # otherwise, add a key,value pair with a count of 1
                token_count_dict.update({token:1})
    sorted_tokens = []
    sorted_counts = []
    # for each token in the dict, iterating from largest to smallest count 
    for w in sorted(token_count_dict, key=token_count_dict.get, reverse=False):
        # add the token to a sorted list of tokens
        sorted_tokens.append(w)
        # add the corresponding count to a list of counts
        sorted_counts.append(token_count_dict[w])    
        print(w, token_count_dict[w])

    # log-scaled bins
    bins = np.logspace(0, 4, 100)
    widths = (bins[1:] - bins[:-1])
    #print(bins)
    #print(widths

    # Calculate histogram
    hist = np.histogram(sorted_counts, bins=bins)
    # normalize by bin width
    hist_norm = hist[0]/widths

    # plot it!
    plt.figure("hist")
    plt.clf()
    plt.bar(bins[:-1], hist[0], widths)
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Token Document Frequency")
    plt.xlabel("# of files a word appears in")
    plt.ylabel("Word count")

    plt.savefig(os.path.join(write_path, "token_document_frequency_histogram"),dpi=500)
   
    return sorted_tokens, sorted_counts


''' PARAMETER: a list of extensions
    DOES: sorts a dictionary with the counts of each extension, 
          writes the top "num_slices" extensions to a file
    RETURNS: a list of sorted tokens and a list of sorted counts '''
def count_and_sort_exts(extensions, num_slices, write_path, dataset_path):
    
    # the name of the top-level directory of the dataset
    dataset_name = path_utilities.get_last_dir_from_path(dataset_path)

    # a dict mapping tokens to the count of how many times they appear
    ext_count_dict = {}
    # for each extension
    for ext in extensions:
        try:
            if ext[2]!="._" and ext[-1]!="~" and ext[0]!="_":
                # if the extension is already in our dict
                if ext in ext_count_dict.keys():
                    # grab the old count
                    old_count = ext_count_dict.get(ext)
                    # increment and update the count in the dict              
                    ext_count_dict.update({ext:old_count+1})
                else:
                    # otherwise, add a key,value pair with a count of 1
                    ext_count_dict.update({ext:1})
        except IndexError:
            if ext.isalnum(): 
                # if the extension is already in our dict
                if ext in ext_count_dict.keys():
                    # grab the old count
                    old_count = ext_count_dict.get(ext)
                    # increment and update the count in the dict              
                    ext_count_dict.update({ext:old_count+1})
                else:
                    # otherwise, add a key,value pair with a count of 1
                    ext_count_dict.update({ext:1})
    sorted_extensions = []
    sorted_counts = []
    # for each extension in the dict, iterating from largest to smallest count 
    for ext in sorted(ext_count_dict, key=ext_count_dict.get, reverse=True):
        # add the extension to a sorted list of extensions
        sorted_extensions.append(ext)
        # add the corresponding count to a list of counts
        sorted_counts.append(ext_count_dict[ext])    
        print(ext, ext_count_dict[ext]) 

    f = open(os.path.join(write_path, "top_exts_" + dataset_name + ".txt" ),'w')
    if (len(sorted_extensions) < num_slices):
        num_slices = len(sorted_extensions)
    for i in range(num_slices):
        f.write(sorted_extensions[i] + "\n")
    f.close()

    return sorted_extensions, sorted_counts

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETERS: a list of extensions, number of pie chart slices, output path
    DOES: creates a pie chart '''
def plot_extension_pie(extensions, num_slices, 
                       write_path, dataset_name):
    sorted_exts, sorted_counts = count_and_sort_exts(extensions, 
                                                     num_slices,
                                                    )
    labels = []
    sizes = []
    for x in range(num_slices):
        labels.append(sorted_exts[x])
        sizes.append(sorted_counts[x])
    plt.figure("pie")
    plt.clf()
    plt.pie(sizes,labels=labels)
    plt.axis('equal')
    plt.title(str(num_slices) + " Most Common Extensions in ")
    pie_path = os.path.join(write_path, "top_exts_pie_" + dataset_name
                            + ".png")
    plt.savefig(pie_path,dpi=300)

#=========1=========2=========3=========4=========5=========6=========7=


def plot_extensions(dataset_path, num_extensions):
    
    allpaths = DFS.DFS(dataset_path) 
    p = Path(os.getcwd()).parent
    dataset_name = get_last_dir_from_path(dataset_path)
    write_path = os.path.join(p, "outputs/", dataset_name + "--output/")
    if not os.path.isdir(write_path):
        os.mkdir(write_path)

    # a list of all the file names (without the paths)
    filenames = []
    for path in allpaths:
        filenames.append(DFS.get_fname_from_path(path))
    filenames_no_ext, exts = remove_all_extensions(filenames) 
    plot_extension_pie(exts, num_extensions, write_path, dataset_name)

    '''
    filenames_path = os.path.join(write_path, "filenames_" 
                                  + dataset_name + ".txt")
    f = open(os.path.join(filenames_path, "w")
    for x in filenames_no_ext:
        f.write(x+ "\n")
        #print(generate_tokens(x))
    f.close()
    sorted_tokens, sorted_counts = count_and_sort_tokens(filenames_no_ext)
    tokens_path = os.path.join(write_path, "tokens_" + dataset_name
                               + ".txt")
    token_file = open(tokens_path, "w")
    for token in sorted_tokens:
        token_file.write(token + "\n")
    token_file.close()
    '''

# MAIN FUNCTION
def main():
    dataset_path = sys.argv[1]
    num_extensions = sys.argv[2]
    plot_extensions(dataset_path, num_extensions)    

if __name__ == "__main__":
    main()
