import os
import argparse
import numpy as np 

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, BertForSequenceClassification

from sklearn.metrics import classification_report


os.environ["WANDB_DISABLED"] = "true"
ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
MULTI_LABEL = False


def load_data(dataset):
    """
    Load training and testing data

    Parameters:
    -----------
        dataset: str
            Either 'toxic' or 'all'

    Returns:
    --------
    datasets.dataset_dict.DatasetDict
        Dataset as HugginFace object
    """
    data_files = {
            "train": os.path.join(ROOT_DIR, "data", dataset, "train.json"), 
            "test": os.path.join(ROOT_DIR, "data", dataset, "test.json")
    }

    return load_dataset('json', data_files=data_files, field="data")


def get_model(model_name):
    """
    Return appropriate name for loading model:
        - "bert" => "bert-base-uncased"
        - "roberta" => "roberta-base"
    
    Will raise error if none of the above

    Parameters:
    -----------
        model_name: str
            Name of model given from user
    
    Returns:
    --------
    str
        Correct name for loading model
    """
    if model_name == "bert":
        return "bert-base-uncased"
    if model_name == "roberta":
        return "roberta-base"
    if "hate" in model_name:
        return "hatebert"
    
    raise ValueError(f"No model of name '{model_name}'")



def compute_metrics(preds_labels):
    """
    Compute the different metrics on predictions

    Returns:
    --------
    dict
       Just so it doesn't throw an error
    """    
    logits, labels = preds_labels

    if MULTI_LABEL:
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        preds = np.where(sigmoid(logits) >= 0.5, 1, 0)

        # Get the report for each individual label
        for c in range(preds.shape[1]): 
            class_preds = preds[:, c]
            class_lbls  = labels[:, c]
            report = classification_report(class_lbls, class_preds)

            print(f"\nReport for Class {c+1}:\n----------------------")
            print(report)
    else:
        preds = np.argmax(logits, axis=-1)
        report = classification_report(labels, preds)
        print(report)

    return {}



def load_hatebert(num_labels):
    """
    Load HateBert from config supplied by authors

    Parameters:
    -----------
        num_labels: int
            Number of labels for datasets
    
    Returns:
    --------
    BertForSequenceClassification
        Bert with HateBert weights
    """
    hatebert_path = os.path.join(ROOT_DIR, "hate_bert")
    
    model = BertForSequenceClassification.from_pretrained(hatebert_path, num_labels=num_labels)
    tokenzier = AutoTokenizer.from_pretrained(hatebert_path)

    return model, tokenzier



def train_model(model_name, data, dataset_name, only_eval, device):
    """
    Train model

    Steps Include:
        1. Load model and tokenizer. Separate for BERT/Roberta and HateBERT
        2. Tokenize Data
        3. Train

    Parameters:
    -----------
        model_name: str
            Name of model
        data: datasets.dataset_dict.DatasetDict
            All data in HugginFace object
        dataset_name: str
            Name of dataset
        only_eval: bool
            Whether to only evaluate a model and not train
        device: str
            Device we are training on 
    
    Returns:
    --------
    """
    num_labels = 2 if dataset_name == "toxic" else 6
    problem_type = "single_label_classification" if num_labels == 2 else "multi_label_classification"

    if model_name == "hatebert":
        model, tokenizer = load_hatebert(num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, problem_type=problem_type)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenzies all train and test data
    token_func = lambda x: tokenizer(x["text"], padding=True, truncation=True) 
    data_tokenized = data.map(token_func, batched=True)

    model = model.to(device)

    training_args = TrainingArguments(
                        output_dir=os.path.join(ROOT_DIR, "checkpoints", model_name, dataset_name),
                        num_train_epochs=5,
                    )
    trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=data_tokenized['train'],
                eval_dataset=data_tokenized['test'],
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )

    if not only_eval:
        trainer.train()

    print(trainer.evaluate())




def main():
    parser = argparse.ArgumentParser(description='Params to run')

    parser.add_argument("--model", help="Model type to run. Either 'bert', 'roberta', or 'hatebert'", type=str, default="bert")
    parser.add_argument("--dataset", help="Either 'all' or 'toxic'", type=str, default="toxic")
    parser.add_argument("--eval", help="Whether to only eval model. By default will train and then evaluate", action="store_true", default=False)
    parser.add_argument("--device", help="Device to run on", type=str, default="cuda")
    
    args = parser.parse_args()

    dataset_name = args.dataset.lower()
    model_name = get_model(args.model.lower())

    # Hack for evaluating model
    if dataset_name == "all":
        global MULTI_LABEL
        MULTI_LABEL = True

    data = load_data(dataset_name)
    train_model(model_name, data, dataset_name, args.eval, args.device)


if __name__ == "__main__":
    main()
