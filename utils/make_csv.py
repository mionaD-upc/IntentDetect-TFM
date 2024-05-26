import os
import csv
import argparse


def process_folders(root_folder):
    output_file_name = os.path.basename(os.path.normpath(root_folder)) + '.csv'
    output_file = os.path.join(root_folder, output_file_name)
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'text', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        id_counter = 1
        
        folder_paths = [os.path.join(root_folder, folder) for folder in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, folder))]
        
        for folder_path in folder_paths:
            label = os.path.basename(folder_path)
            file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.txt')]
            for file_path in file_paths:
                with open(file_path, 'r', encoding='utf-8') as txtfile:
                    text = txtfile.read()
                    writer.writerow({'id': f'{id_counter}', 'text': text, 'label': label})
                    id_counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to the training data folder")
    args = parser.parse_args()

    root_folder = args.path
    process_folders(root_folder)
