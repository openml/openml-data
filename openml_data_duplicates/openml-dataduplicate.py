import arff
from collections import defaultdict
import logging
import openml
import os
import pandas as pd
import sys
import numpy as np


def load_arff(file_path):
    with open(file_path, 'r') as fh:
        return arff.load(fh)


def get_metafeatures(data):
    name = data['relation']
    instances = len(data['data'])
    features = len(data['data'][0])
    missing = len([v for row in data['data'] for v in row if v is None])
    return name, instances, features, missing

def compare(data_features, characteristics):
    """ Compares `characteristics` to the `data`.
    :param data: dict. arff data
    :param characteristics: dict. OpenML dataset description
    :return: Tuple[bool, int, int, int].
             True if dataset name of A is contained in B or vice versa.
             Difference in number of samples.
             Difference in number of features.
             Difference in number of missing values.
    """
    name, instances, features, missing = data_features
    return (name.lower() in characteristics['name'].lower() or characteristics['name'].lower() in name.lower(),
            abs(characteristics.get('NumberOfInstances', float('nan')) - instances),
            abs(characteristics.get('NumberOfFeatures', float('nan')) - features),
            abs(characteristics.get('NumberOfMissingValues', float('nan')) - missing))

def create_df_matches(datasets_check=None):
    oml_datasets = openml.datasets.list_datasets()
    comparisons = []
    new_datasets = openml.datasets.get_datasets(datasets_check, download_data=False)
    for i, data in enumerate(new_datasets):
        file_path = data.id
        logging.info("[{:3d}/{:3d}] {}".format(i+1, len(datasets_check), data.name))
        
        new_data = oml_datasets.get(file_path)
        new_data_metafeatures = [new_data.get(k) for k in ('name','NumberOfInstances','NumberOfFeatures','NumberOfMissingValues')]
        
        for did, oml_dataset in oml_datasets.items():
#            Uncomment this if you do not want the dataset id as a duplicate of itself
#            if did == new_data.get('did'):
#                continue
            name_match, d_instances, d_features, d_missing = compare(new_data_metafeatures, oml_dataset)
            if name_match or sum([d_instances, d_features, d_missing]) == 0:
                comparisons.append([data.id, data.name, oml_dataset.get('name'), did, name_match, d_instances, d_features, d_missing])
    return pd.DataFrame(comparisons, columns=['did', 'name', 'name_duplicate', 'did_duplicate', 'name_match', 'd_instances', 'd_features', 'd_missing'])

def move_bad_files(folder):
    sub_folder = 'bad/'
    arff_files = [filepath for filepath in os.listdir(folder) if filepath.endswith('.arff')]
    
    if not os.path.exists(os.path.join(folder, sub_folder)):
        os.makedirs(os.path.join(folder, sub_folder))
        
    for i, file_path in enumerate(arff_files):
        try:
            load_arff(os.path.join(folder, file_path))
        except arff.ArffException as e:
            logging.info("[{:3d}/{:3d}] Moving {}, reason: {}".format(i+1, len(arff_files), file_path[:-5], str(e)))
            os.rename(os.path.join(folder, file_path), os.path.join(folder + 'bad/', file_path))

def get_matches_per_dataset(df, fn, exclude=[]):
    matches = defaultdict(list)
    for i, row in df.iterrows():
        if row['did_duplicate'] in exclude:
            continue
        if fn(row):
            matches[row['name']].append(row['did_duplicate'])
    return matches

def row_print_dict(d, df):
    if len(d) == 0:
        print("[empty]")
        return
    max_len = max([len(k) for k in d])
    for k, values in d.items():
        print('({}){}: {}'.format(int(df[df.name==k].did[:1]), k.ljust(max_len), values))

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig()

    logging.info("Checking for matches against OpenML.")

    # Put here the datasets to look for duplicates (this is just an example)
    source_datasets = openml.study.get_suite(271).data
    df = create_df_matches(source_datasets)

    nan = np.nan

    logging.info("Aggregating results...")

#    Uncomment this to show all the duplicates found regardless of the criteria
#    print("The following duplicates were found in total:")
#    all_matches = dict()
#    for name,group in df.groupby(['name']):
#        all_matches[name] = list(group.did_duplicate)
#    row_print_dict(all_matches,df)

    def combine_lists(lists):
        return [y for x in lists.values() for y in x]

    matched_datasets = []

    def perfect_match(row): 
        return row['name_match'] and row['d_instances'] == 0 and row['d_features'] == 0 and row['d_missing'] == 0   

    perfect_matches = get_matches_per_dataset(df, fn=perfect_match, exclude=matched_datasets)
    matched_datasets += combine_lists(perfect_matches)
    print("The following datasets have matching names (A contained in B or B contained in A), and have the same number of instances, features and missing values:")
    row_print_dict(perfect_matches, df)

    def close_match(row):
        return row['name_match'] and sum([row['d_instances'] == 0, row['d_features'] == 0, row['d_missing'] == 0]) == 2

    close_matches = get_matches_per_dataset(df, fn=close_match, exclude=matched_datasets)
    matched_datasets += combine_lists(close_matches)
    print("The following datasets have matching names, but differ in either instances, features, or missing values:")
    row_print_dict(close_matches, df)

    def name_match(row):
        return row['name_match'] and sum([row['d_instances'] == 0, row['d_features'] == 0, row['d_missing'] == 0]) < 2

    name_matches = get_matches_per_dataset(df, fn=name_match, exclude=matched_datasets)
    matched_datasets += combine_lists(name_matches)
    print("The following datasets have matching names, but differ in more than one way:")
    row_print_dict(name_matches, df)

    def shape_match(row):
        return (not row['name_match']) and row['d_instances'] == 0 and row['d_features'] == 0 and row['d_missing'] == 0

    shape_matches = get_matches_per_dataset(df, fn=shape_match)
    matched_datasets += combine_lists(shape_matches)
    print("The following datasets do not have matching names,"
          "but have the same number of instances, features and missing values:")
    row_print_dict(shape_matches, df)

    all_datasets = df['did']
    no_matches = [did for did in all_datasets if did not in matched_datasets]
    print("The following datasets do not match any of the above criteria:")
    for no_match in no_matches:
        print(no_match)