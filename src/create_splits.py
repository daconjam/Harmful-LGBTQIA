import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")



def split_toxic_data():
    """
    Split the data with the toxic label into training and testing set with 80/20 split 
    
    Stratified to account for imbalance

    Store results in data/toxic directory in json format conducive for HuggingFace
    """
    data = pd.read_csv(os.path.join(DATA_DIR, "full_data.csv"), index_col=0)
    X, Y = data['comments'], data['toxicity']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    train_data = [{"text": x, "label": int(y)} for x, y in zip(x_train, y_train)]
    test_data = [{"text": x, "label": int(y)} for x, y in zip(x_test, y_test)]

    print(">>>Saving toxic data")
    print(f">>> Train: {len(train_data)},  Test: {len(test_data)} samples")

    with open(os.path.join(DATA_DIR, "toxic", "train.json"), "w") as f:
        json.dump({"data": train_data}, f, indent=4)

    with open(os.path.join(DATA_DIR, "toxic", "test.json"), "w") as f:
        json.dump({"data": test_data}, f, indent=4)



def split_all_data():
    """
    Split the data with all labels into training and testing set with 80/20 split.

    Label is a list of all possible labels (e.g. [1, 0, 0, 1, 1]) 

    Store results in data/all directory in json format conducive for HuggingFace
    """
    cols = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']

    data = pd.read_csv(os.path.join(DATA_DIR, "full_data.csv"), index_col=0)
    data['all_lbl'] = data.apply(lambda x: [x[c] for c in cols], axis=1)

    X, Y = data['comments'], data['all_lbl']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    train_data = [{"text": x, "label": y} for x, y in zip(x_train, y_train)]
    test_data = [{"text": x, "label": y} for x, y in zip(x_test, y_test)]

    print(">>>Saving all data")
    print(f">>> Train: {len(train_data)},  Test: {len(test_data)} samples")

    with open(os.path.join(DATA_DIR, "all", "train.json"), "w") as f:
        json.dump({"data": train_data}, f, indent=4)

    with open(os.path.join(DATA_DIR, "all", "test.json"), "w") as f:
        json.dump({"data": test_data}, f, indent=4)


def main():
    split_toxic_data()
    split_all_data()


if __name__ == "__main__":
    main()
