import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

def predict_with_trepan(row):
    # Features: 'worst perimeter', 'worst area', 'worst concave points', 'worst radius'
    wp = row['worst perimeter']
    wa = row['worst area']
    wcp = row['worst concave points']
    wr = row['worst radius']

    # Rule 1
    if (wp <= 115.655) and (wa <= 739.887) and (wcp <= 0.142):
        return 1
    # Rule 2
    elif (wp <= 115.655) and (wa <= 739.887) and (wcp > 0.142) and (wcp <= 0.173):
        return 1
    # Rule 3
    elif (wp <= 115.655) and (wa <= 739.887) and (wcp > 0.142) and (wcp > 0.173):
        return 0
    # Rule 4
    elif (wp <= 115.655) and (wa > 739.887) and (wa <= 896.112) and (wp <= 105.95):
        return 1
    # Rule 5
    elif (wp <= 115.655) and (wa > 739.887) and (wa <= 896.112) and (wp > 105.95) and (wr <= 15.977):
        return 0
    # Rule 6
    elif (wp <= 115.655) and (wa > 739.887) and (wa <= 896.112) and (wp > 105.95) and (wr > 15.977) and (wr <= 16.319) and (wcp <= 0.145):
        return 1
    # Rule 7
    elif (wp <= 115.655) and (wa > 739.887) and (wa <= 896.112) and (wp > 105.95) and (wr > 15.977) and (wr <= 16.319) and (wcp > 0.145):
        return 0
    # Rule 8
    elif (wp <= 115.655) and (wa > 739.887) and (wa <= 896.112) and (wp > 105.95) and (wr > 15.977) and (wr > 16.319):
        return 1
    # Rule 9
    elif (wp <= 115.655) and (wa > 739.887) and (wa > 896.112):
        return 0
    # Rule 10
    elif (wp > 115.655) and (wr <= 17.574):
        return 0
    # Rule 11
    elif (wp > 115.655) and (wr > 17.574):
        return 0
    else:
        return None  # No rule matches

# Test the function
if __name__ == "__main__":
    frame = load_breast_cancer(as_frame=True)
    df = frame['data']
    
    # Select only the features used in rules
    features_used = ['worst perimeter', 'worst area', 'worst concave points', 'worst radius']
    df_subset = df[features_used]
    
    predictions = df_subset.apply(predict_with_trepan, axis=1)
    print("Predictions for first 10 samples:")
    print(predictions.head(10))
    print("\nNumber of samples without a rule match:", predictions.isna().sum())
