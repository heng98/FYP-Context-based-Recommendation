# mkdir Dataset
# cd Dataset

# wget http://tangra.cs.yale.edu/newaan/data/releases/2014/aanrelease2014.tar.gz
# tar -zxvf aanrelease2014.tar.gz
# rm aanrelease2014.tar.gz

# cd ..

python data/process_ann_data.py --save_path processed_aan_data --dataset_path Dataset/aan/papers_text