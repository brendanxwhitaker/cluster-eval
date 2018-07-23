import argparse
import pprint
import sys

#=========1=========2=========3=========4=========5=========6=========7=

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument('--dataset_path',
            type=str,
            default='')
    argparser.add_argument('--plot_extensions',
            type=str,
            default='n')
    argparser.add_argument('--convert',
            type=str,
            default='n') 
    argparser.add_argument('--cluster_struct',
            type=str,
            default='y')
    argparser.add_argument('--cluster_text',
            type=str,
            default='y')
    argparser.add_argument('--num_extensions',
            type=int,
            default=15)
    argparser.add_argument('--fill_threshold',
            type=int,
            default=0.4)
    argparser.add_argument('--overwrite_distmat_struct',
            type=str,
            default='n')
    argparser.add_argument('--overwrite_plot_struct',
            type=str,
            default='n')
    argparser.add_argument('--overwrite_tokens_text',
            type=str,
            default='n')
    argparser.add_argument('--overwrite_clusters_text',
            type=str,
            default='n')
    argparser.add_argument('--num_clusters_start',
            type=int,
            default=10)
    argparser.add_argument('--num_clusters_end',
            type=int,
            default=0)
    args = argparser.parse_args()

    print('------------------------------------------------')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print('------------------------------------------------')

    return args
