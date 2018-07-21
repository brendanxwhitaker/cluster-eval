import os

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: a single string
    RETURNS: the string with all / replaced with @ '''
def str_encode(string):
    return string.replace("/","@")

''' PARAMETER: a single string
    RETURNS: the string with all @ replaced with / '''
def str_decode(string):
    return string.replace("@","/")

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: a full path
    RETURNS: only the filename '''
def get_fname_from_path(path):    
    filename = ""
    for c in path[::-1]:
        if c=="/" or c=="@":
            break
        filename = c+filename
    return filename

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: a single filename 
    RETURNS: the file without its extension '''
def remove_extension(filename):    
    length = len(filename)
    for ch in filename[::-1]:
        if ch==".":
            break
        length = length - 1 
    return filename[:length-1]

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: a single filename. 
    RETURNS: the file extension from that filename. '''
def get_single_extension(filename):    
    extension = ""
    length = len(filename)
    for ch in filename[::-1]:
        if ch==".":
            break
        # "extension" contains just the extension from a filename
        extension = ch + extension
        length = length - 1 
    return extension.lower()

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: a directory path
    RETURNS: a list of the immediate children of that directory '''
def get_immediate_subdirectories(a_dir):
    sub_list = []
    for name in os.listdir(a_dir):
        if os.path.isdir(os.path.join(a_dir, name)):
            sub_list.append(os.path.join(a_dir, name)+"/")
    return sub_list

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: "char" a character to find
               "string" a string to find the character in
    RETURNS: index of "char" in "string". Returns length of "string"
             if there are no instances of that character. '''
def find_char_from_end(string, char):    
        length = len(string)
        # we iterate on the characters starting from end of the string
        pos = length - 1
        # dot pos will be position of the first period from the end,
        # i.e. the file extension dot. If it is still "length" at end,
        # we know that there are no dots in the filename. 
        dot_pos = length
        while (pos >= 0):
            if (filename[pos] == char):
                dot_pos = pos
                break
            pos = pos - 1
        return dot_pos

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: a string with a path 
    RETURNS: "/hello/no/" from argument "/hello/no/yes/"
    NOTE: works with or without the trailing "/" '''
def remove_path_end(path):
    # remove trailing "/" if it exists. 
    if (path[len(path) - 1] == "/"):
        path = path[0:len(path) - 1]

    # location of the delimiter "/"
    delim_loc = 0
    j = len(path) - 1
    # iterating to find location of first "/" from right
    while (j >= 0):
        if (path[j] == "/"):
            delim_loc = j
            break
        j = j - 1
    # shorten the path, include ending "/"
    shortened_path = path[0:delim_loc + 1]
    return shortened_path

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: a string with a directory
    RETURNS: only the last folder name '''
def get_last_dir_from_path(path):    
    filename = ""
    if path[len(path) - 1] == "/" or path[len(path) - 1] == "@":
        path = path[0:len(path) - 1]
    for c in path[::-1]:
        if c=="/" or c=="@":
            break
        filename = c+filename
    return filename

#=========1=========2=========3=========4=========5=========6=========7=

def main():
    print("This file is just for importing functions, don't run it. ")    

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main() 
