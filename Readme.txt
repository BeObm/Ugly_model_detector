
0. Open Anaconda teminal and navigate to code source path


A- To test the model on new images:

1- put all the images in a folder. The folder may contains nested subfolder, it is ok.
2. run the following command:
pyton test_model.py --test_path=[path to images hee] --output_path=[output_Excelfile_for_result]

B- to copy ugly model or good model from the result file to a specific folder
  run the following command:
  move_model_pic.py --excel_file_path=[path to Excelfile] --type=[o for good, 1 for ugly] --destination_folder=[path destination_folder]

Note: if possible, the destination path will be created if not exist.