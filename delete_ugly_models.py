import os
from multiprocessing import Pool
import pandas as pd
def delete_txt_files(file_path):
    try:
        if file_path.endswith('.txt'):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

def delete_txt_files_in_folder(folder_path):
    try:
        with Pool() as pool:
            file_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]
            pool.map(delete_txt_files, file_paths)
    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")



def delete_images_based_on_excel(excel_file_path):
    df = pd.read_excel(excel_file_path)

    rows_to_delete = df[df['Predicted Label'] == 1]

    for file_path in rows_to_delete['File Path']:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


excel_file_path = 'predictions.xlsx'
delete_images_based_on_excel(excel_file_path)

