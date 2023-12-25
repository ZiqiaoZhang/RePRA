from Computing import *
from Utils import *
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
ClassificationTaskNames = [
    'BACE',
    'BBBP',
    'ClinTox',
    'SIDER',
    'Tox21'
]
RegressionTaskNames = [
    'ESOL',
    'FreeSolv',
    'Lipo',
    'QM7',
    'QM8'
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
    'Zmetric': 'MinMaxEud',
    'Fd2s': 'Linear',
    'Ymetric': 'MinMaxEud',
}

for PretrainModel in PretrainModelNames:
    for TargetTask in TargetTaskNames:
        print(f"Calculating {PretrainModel} on {TargetTask}.")
        Settings.update({'PretrainModel':PretrainModel})
        Settings.update({'TargetTask': TargetTask})

        # before calculating, we have:
        # Eud(z1,z2), Eud(y1,y2), CosineSim(z1,z2)
        # Calculate Sim_Z
        filename = FileNames(Settings, 'SimArray')
        if not CheckFileExists(filename):
            print("Calculating Sim_Z")
            SimArray = CalSimArray(Settings)
            with open(filename, 'w') as f:
                json.dump(SimArray, f)
        else:
            print("Loading Sim_Z")
            SimArray = LoadFile(filename)

        # Load Diff_Y
        filename = FileNames(Settings, 'DiffArray')
        if not CheckFileExists(filename):
            raise FileExistsError(f"Cannot find DiffArray file of {TargetTask} dataset.")
        else:
            print("Loading Diff_Y")
            DiffArray = LoadFile(filename)


        # Calculate Delta and epsilon
        # if Classification
        filename = FileNames(Settings, 'Thresholds')
        if not CheckFileExists(filename):
            print(f"Calculating Thresholds of {PretrainModel} on {TargetTask}.")
            if TargetTask in ClassificationTaskNames:
                filename_epsilon = FileNames(Settings,'Epsilon')
                if not CheckFileExists(filename_epsilon):
                    print(f"Calculating epsilon.")
                    epsilon = ClassificationTaskEpsilonCalculating(Settings)
                    with open(filename_epsilon, 'w') as f:
                        json.dump(epsilon, f)
                else:
                    epsilon = LoadFile(filename_epsilon)
                delta_AC, delta_SH = ClassificatinoTaskDeltaCalculating(Settings)
                thresholds = {'epsilon': epsilon, 'delta_AC': delta_AC, 'delta_SH': delta_SH}
            else:
                (delta_AC, epsilon_AC), (delta_SH, epsilon_SH) = RegressionTaskDeltaEpsilonCalculating(Settings)
                thresholds = {'delta_AC': delta_AC, 'delta_SH': delta_SH,'epsilon_AC': epsilon_AC, 'epsilon_SH': epsilon_SH}
            with open(filename, 'w') as f:
                json.dump(thresholds, f)
        else:
            print(f"Loading thresholds.")
            thresholds = LoadFile(filename)
        print(f"Thresholds: {thresholds}")
        Settings.update({'Thresholds': thresholds})

        # Draw RPSMap
        filename = FileNames(Settings, 'RPSMap')
        if not CheckFileExists(filename):
            print(f"Drawing RPSMaps")
            draw_scatter_new(SimArray, DiffArray, Settings)


        ############################
        # if MorganFP, Calculate the C1 and C4, then continue the loop.
        if PretrainModel == 'MorganFP':
            filename = FileNames(Settings, 'MorganFPC1C4')
            if not CheckFileExists(filename):
                print(f"Calculating C1 C4 of MorganFP on {TargetTask} dataset.")
                C1C4 = C1C4Calculating_Array(SimArray, DiffArray, Settings)
                with open(filename, 'w') as f:
                    json.dump(C1C4,f)
        else:
            # Calculate M1
            # s_{AD}
            filename = FileNames(Settings, 'M1Array')
            if not CheckFileExists(filename):
                print(f"Calculating M1 value of {PretrainModel} on {TargetTask} dataset.")
                M1 = Metric1Calculating_Array(SimArray, DiffArray, Settings)
                with open(filename, 'w') as f:
                    json.dump(M1, f)

        #     # Calculate M2
        #     filename = FileNames(Settings, 'M2Array')
        #     if not CheckFileExists(filename):
        #         print(f"Calculating M2 value of {PretrainModel} on {TargetTask} dataset.")
        #         M2 = Metric2Calculating_Array(SimArray, DiffArray, Settings)
        #         with open(filename, 'w') as f:
        #             json.dump(M2, f)
        #
            # Calculate M3
            # s_{IR}
            filename = FileNames(Settings, 'M3Array')
            if not CheckFileExists(filename):
                print(f"Calculating M3 value of {PretrainModel} on {TargetTask} dataset.")
                M3 = Metric3Calculating_Array(SimArray, DiffArray, Settings)
                with open(filename, 'w') as f:
                    json.dump(M3, f)
        
            filename = FileNames(Settings, 'M3C1C4')
            if not CheckFileExists(filename):
                print(f"Calculating M3C1C4 value of {PretrainModel} on {TargetTask} dataset.")
                M3C1C4 = Metric3Calculating_Array_C1C4(SimArray, DiffArray, Settings)
                with open(filename, 'w') as f:
                    json.dump(M3C1C4, f)




