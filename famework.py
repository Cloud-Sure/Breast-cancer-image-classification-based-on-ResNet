import os
import shutil

# Raw data folder path
source_dir = r'D:\data\BreaKHis_v1\histology_slides\breast\malignant\SOB'

# Destination data folder path
target_dir = r'D:\data\BreaKHis\histology_slides\breast\malignant'

def reorganize_structure(source_dir, target_dir):
    # Iterate through the subfolders under the SOB folder
    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)

        if os.path.isdir(category_path):
            # Traverse the sample number folder under that category
            for sample in os.listdir(category_path):
                sample_path = os.path.join(category_path, sample)

                if os.path.isdir(sample_path):
                    # Traverse the magnification folder for each sample
                    for magnification in os.listdir(sample_path):
                        magnification_path = os.path.join(sample_path, magnification)

                        if os.path.isdir(magnification_path):
                            # Create a folder under the target directory by pressing the magnification
                            target_magnification_path = os.path.join(target_dir, magnification)
                            if not os.path.exists(target_magnification_path):
                                os.makedirs(target_magnification_path)

                            # Create a folder for each category under the Target Multiple folder
                            target_category_path = os.path.join(target_magnification_path, category)
                            if not os.path.exists(target_category_path):
                                os.makedirs(target_category_path)

                            # Move files under this multiple to the corresponding category folder in the destination folder
                            for file_name in os.listdir(magnification_path):
                                file_path = os.path.join(magnification_path, file_name)
                                if os.path.isfile(file_path):
                                    shutil.move(file_path, os.path.join(target_category_path, file_name))


def main():
    reorganize_structure(source_dir, target_dir)
    print("Folder structure has been reorganized!")


if __name__ == '__main__':
    main()
