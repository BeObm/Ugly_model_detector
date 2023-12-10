from shutil import copy2
import os
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import argparse

def multi_copy(args):
    src_path, destination_folder = args
    try:
        copy2(src_path, destination_folder)
        print(f"Copied: {src_path} to {destination_folder}")
    except:
        print(f"Error copying {src_path}: {e}")


def copy_file(excel_file_path, destination_folder, num_proc=8, type_image=0):
    df = pd.read_excel(excel_file_path)
    if type_image == 0:
        row_to_copy = df[df["Predicted Label"] == 0]
    elif type_image == 1:
        row_to_copy = df[df["Predicted Label"] == 1]

    file_path_to_copy = row_to_copy["File Path"].tolist()

    arg_list = [(src_path, destination_folder) for src_path in file_path_to_copy]

    with Pool(num_proc) as pool:
        pool.map(multi_copy,arg_list)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel_file_path", help="path to the excel file", default="C:/Users/Orca-Epyc7B13/Desktop/Test/predictions_results_test_sample.xlsx")
    parser.add_argument("--type", type= int, help="0 for good models 1 for ugly models", default=1, choices=[0,1])
    parser.add_argument("--destination_folder", help="destination_folder", default='C:/Users/Orca-Epyc7B13/Desktop/Test/ugly')
    args = parser.parse_args()
    os.makedirs(args.destination_folder,exist_ok=True)

    # test_loader = prepare_test_dataset(args.test_path)
    copy_file(args.excel_file_path, args.destination_folder, num_proc=8, type_image=args.type)
