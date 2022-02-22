import pandas as pd
import numpy as np

raw_csv = pd.read_csv('./raw-data-split/train.csv')

list(raw_csv.columns.values)[1:]
 

def csv_preprocess(raw_csv) -> pd.DataFrame:
    
    if type(raw_csv) == str:
        raw_csv = pd.read_csv(raw_csv)
        
    
    raw_csv = raw_csv.replace({np.nan: 0, '1.0': 1, '0': 0})
    
    raw_csv = raw_csv.loc[raw_csv['Frontal/Lateral'] == 'Frontal']
    
    raw_csv = raw_csv.drop(labels = ['Enlarged Cardiomediastinum', 
                                       'Lung Opacity',
                                       'Pneumonia',
                                       'Pleural Other',
                                       'No Finding',
                                       'Support Devices',
                                       'Sex',
                                       'Age',
                                       'AP/PA',
                                       'Sex',
                                       'Age',
                                       'Frontal/Lateral'
                                       ],
                             axis='columns'
                             )
    
    raw_csv.replace({'Cardiomegaly': {-1: 0},
                      'Consolidation': {-1: 0},
                      'Atelectasis': {-1: 1},
                      'Edema': {-1: 1},
                      'Pleural Effusion': {-1: 1},
                      'Pneumothorax': {-1: 0},
                      'Lung Lesion': {-1: 1},
                      'Fracture': {-1: 0}
                      }, inplace=True)
    
    raw_csv.iloc[:,1:] = raw_csv.iloc[:,1:].astype('int8')
    
    assert raw_csv.iloc[:,1:].values.any() >= 0
    
    return raw_csv
