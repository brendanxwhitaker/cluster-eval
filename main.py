from options import load_arguments
import converting_utilities
import document_clustering
import schema_clustering
import tokenizer
import sys
import os

# EVERY OTHER FUNCTION WILL BE CALLED FROM HERE.

#=========1=========2=========3=========4=========5=========6=========7=

def main():
    
    print("ARGUMENTS: ")
    args = load_arguments()
    print("Arguments loaded. ")
   
    # num_clusters_text
    # overwrite_tokens

    if args.reconvert == 'y' or args.reconvert == 'Y':
        tokenizer.plot_extensions(
                                  args.dataset_path,
                                  args.num_extensions,
                                 )  
    if args.reconvert == 'y' or args.reconvert == 'Y':
        converting_utilities.convert(
                                     args.dataset_path, 
                                     args.num_extensions, 
                                    ) 
    if args.cluster_tables == 'y' or args.cluster_tables == 'Y':
        schema_clustering.runflow(
                                  args.dataset_path, 
                                  args.num_clusters_tabular, 
                                  args.overwrite_distmat_tabular, 
                                  args.overwrite_plot_tabular,
                                  args.fill_threshold,
                                 )
    if args.cluster_text == 'y' or args.cluster_text == 'Y':
        document_clustering.runflow(args.num_clusters_text, 
                                    args.overwrite_tokens_text,
                                    args.overwrite_clusters_text,
                                    args.dataset_path)
    
 
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
