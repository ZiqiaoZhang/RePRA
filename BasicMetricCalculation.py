import os
from Utils import *
from Computing import *
import json

TargetTaskNames = [
    'BACE',
    'BBBP',
    'ClinTox',
    'ESOL',
    'FreeSolv',
    'Lipo',
    'QM7',
    'QM8',
    'SIDER',
    'Tox21',
]
PretrainModelNames = [
    'MorganFP',
    'GROVER',
    'ChemBert',
    'GraphLoG',
    'MAT',
    'SMILESTransformer',
    'PretrainGNNs',
    'Pretrain8'
]
Settings = {
    'RootPath':'~/codes/MetricForFeatureSpace/',
    'CodeVersion': 'MetricExpCodes',
}

for TargetTask in TargetTaskNames:
    Settings.update({"TargetTask": TargetTask})

    # Load Properties
    filename = FileNames(Settings, 'Property')
    if not CheckFileExists(filename):
        print(f"Extracting Properties of {TargetTask} from Dataset.")
        property_extractor = TaskPropertyExtractor()
        property_extractor.PropertyExtract(TargetTask)
        Properties = LoadFile(filename)
    else:
        print(f"Loading Properties of {TargetTask}")
        Properties = LoadFile(filename)

    # F: Properties: [task_num, total_num]

    # Calculate DiffArray   d(y1,y2)
    filename = FileNames(Settings, 'DiffArray')
    if not CheckFileExists(filename):
        print(f"Calculating the MinMaxEud of Properties of {TargetTask} dataset.")
        if (TargetTask != 'Tox21') and (TargetTask != 'Toxcast'):
            DiffArray = CalDiffArray(Properties)
        else:
            Properties = FillMissingValueByDefault(Properties)
            DiffArray = CalDiffArray(Properties)
        with open(filename, 'w') as f:
            json.dump(DiffArray, f)


    # Calculate PretrainModels on TargetTasks
    import time
    start_time = time.time()
    for PretrainModel in PretrainModelNames:
        Settings.update({'PretrainModel': PretrainModel})

        # Load Feature
        filename = FileNames(Settings, 'Feature')
        if not CheckFileExists(filename):
            raise FileExistsError(f"Feature of {PretrainModel} on {TargetTask} have not been extracted!")
        else:
            print(f"Loading features.")
            Features = LoadFile(filename)

        # Z: Features: [total_num, feature_num]

        # Calculate MinMaxEud of Features of PretrainModel on TargetTask
        # d(z1,z2)
        filename = FileNames(Settings, 'FeatureMinMaxEud')
        if not CheckFileExists(filename):
            print(f"Calculating the MinMaxEud of features of {PretrainModel} on {TargetTask} dataset.")
            DistArray = CalMinMaxEuclidean(Features)
            # print(DistArray)
            with open(filename, 'w') as f:
                json.dump(DistArray, f)

        # Calculate CosineSim of Features of PretrainModel on TargetTask
        filename = FileNames(Settings, 'FeatureCosineSim')
        if not CheckFileExists(filename):
            print(f"Calculating the CosineSim of features of {PretrainModel} on {TargetTask} dataset.")
            SimArray = CalCosineSimArray(Features)
            with open(filename, 'w') as f:
                json.dump(SimArray, f)
    end_time = time.time()
    print(f"Total time for calculating arrays is {end_time - start_time}")

