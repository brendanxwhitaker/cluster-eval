import sys
import os
import hashlib

def chunk_reader(fobj, chunk_size=1024):
    """Generator that reads a file in chunks of bytes"""
    while True:
        chunk = fobj.read(chunk_size)
        if not chunk:
            return
        yield chunk

def check_for_duplicates(paths, hash=hashlib.sha1):
    num_hidden_dupl = 0
    num_dupl = 0
    hashes = {}
    for path in paths:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                hashobj = hash()
                for chunk in chunk_reader(open(full_path, 'rb')):
                    hashobj.update(chunk)
                file_id = (hashobj.digest(), os.path.getsize(full_path))
                duplicate = hashes.get(file_id, None)
                if duplicate:
                    print ("Duplicate found: %s and %s" % (full_path, duplicate))
                    if (filename[0] == "."):
                        num_hidden_dupl = num_hidden_dupl + 1
                    else:
                        num_dupl = num_dupl + 1
                else:
                    hashes[file_id] = full_path

    print("============================")
    print("Found " + str(num_dupl) + " duplicates. ")
    print("Found " + str(num_hidden_dupl) + " hidden duplicates. ")
    print("============================")

if sys.argv[1:]:
    check_for_duplicates(sys.argv[1:])
else:
    print ("Please pass the paths to check as parameters to the script")
