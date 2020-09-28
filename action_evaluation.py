"""Script evaluates action prediction along with attributes.

Author(s): Satwik Kottur
"""


from absl import app, flags
import collections
import json
import argparse
import os
import numpy as np
import ipdb
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "action_json_path", "data/furniture_api_calls.json", "Ground truth API calls"
)
flags.DEFINE_string(
    "model_output_path", None, "Action API predictions by the model"
)


IGNORE_ATTRIBUTES = [
    "minPrice",
    "maxPrice",
    "furniture_id",
    "material",
    "decorStyle",
    "intendedRoom",
    "raw_matches",
    "focus"  # fashion
]


def evaluate_action_prediction2(gt_actions, model_actions, single_round_eval=False):
    """Evaluates action prediction using the raw data and model predictions.
    Args:
        gt_actions: Ground truth actions + action attributes
        model_actions: Actions + attributes predicted by the model
        single_round_eval: Evaluate only for the last turn.
    """
    gt_actions_pool = {ii["dialog_id"]: ii for ii in gt_actions}
    matches = {"action": [], "attributes": [], "perplexity": []}
    confusion_dict = collections.defaultdict(list)
    skipped = 0
    for model_datum in model_actions:
        dialog_id = model_datum["dialog_id"]
        num_gt_rounds = len(gt_actions_pool[dialog_id]["actions"])
        for round_datum in model_datum["predictions"]:
            round_id = round_datum["turn_id"]
            # Skip if single_round_eval and this is not the last round.
            if single_round_eval and round_id != num_gt_rounds - 1:
                continue

            gt_datum = gt_actions_pool[dialog_id]["actions"][round_id]
            action_match = gt_datum["action"] == round_datum["action"]

            # Record matches and confusion.
            matches["action"].append(action_match)
            # matches["perplexity"].append(
                # round_datum["action_log_prob"][gt_datum["action"]]
            # )
            confusion_dict[gt_datum["action"]].append(round_datum["action"])
      
            # Get supervision for action attributes.
            supervision = gt_datum["action_supervision"]

            if supervision is not None and "args" in supervision:
                supervision = supervision["args"]
            if supervision is None:
                skipped += 1
                continue
            # Case 1: Action mismatch -- record False for all attributes.
            if not action_match:
                print("Action Wrong for dialog_id {} , round_id {} \n Right Action : {} \n predicted Action {} \n".format(dialog_id,round_id,gt_datum["action"],round_datum["action"]))
                for key in supervision.keys():
                    if key in IGNORE_ATTRIBUTES:
                        continue
                    matches["attributes"].append(False)
            # Case 2: Action matches -- use model predictions for attributes.
            else:
                # ipdb.set_trace()
                for key in supervision.keys():
                    if key in IGNORE_ATTRIBUTES:
                        continue
                    gt_key_vals = supervision[key]
                    model_key_vals = round_datum["attributes"][key]
                    if not len(gt_key_vals):
                        continue
                    # For fashion, this is a list -- multi label prediction.
                    if isinstance(gt_key_vals, list):
                        assert isinstance(model_key_vals, list), (
                            "Model should also predict a list for attributes"
                        )
                        recall = np.mean(
                            [(ii in model_key_vals) for ii in gt_key_vals]
                        )
                        if len(model_key_vals):
                            precision = np.mean(
                                [(ii in gt_key_vals) for ii in model_key_vals]
                            )
                        else:
                            precision = 0

                        f1_score = (2 * recall * precision) / (recall + precision + 1e-5)
                        if f1_score < 0.9 : 
                            print("\nAttribute Wrong => Dialog_id : {}, turn_id: {}\n Action_Ans : {} / Action_Predicted  : {} \n Attribute answer {} , prediction {}".format(dialog_id,round_id,gt_datum["action"],round_datum["action"],gt_key_vals, model_key_vals))
                        matches["attributes"].append(f1_score)
                    else:
                        # For furniture, this is a string -- single label prediction.
                        matches["attributes"].append(gt_key_vals == model_key_vals)

    print("#Instances evaluated API: {}".format(len(matches["action"])))
    print("skipped {}".format(skipped))
    return {
        "action_accuracy": np.mean(matches["action"]),
        "attribute_accuracy": np.mean(matches["attributes"])
    }
    #"action_perplexity": np.exp(-1 * np.mean(matches["perplexity"])),


def evaluate_action_prediction(gt_actions, model_actions):
    """Evaluates action prediction using the raw data and model predictions.

    Args:
        gt_actions: Ground truth actions + action attributes
        model_actions: Actions + attributes predicted by the model
    """

    gt_actions_pool = {ii["dialog_id"]: ii for ii in gt_actions}
    matches = {"action": [], "attributes": [], "perplexity": []}
    confusion_dict = collections.defaultdict(list)

    for model_datum in model_actions:
        dialog_id = model_datum["dialog_id"]
        for round_id, round_datum in enumerate(model_datum["predictions"]):
            gt_datum = gt_actions_pool[dialog_id]["actions"][round_id]
            action_match = gt_datum["action"] == round_datum["action"]
            # Record matches and confusion.
            matches["action"].append(action_match)
            '''
            matches["perplexity"].append(
                round_datum["action_log_prob"][gt_datum["action"]]
            )
            '''
            confusion_dict[gt_datum["action"]].append(round_datum["action"])

            # Get supervision for action attributes.
            supervision = gt_datum["action_supervision"]
            if supervision is not None and "args" in supervision:
                supervision = supervision["args"]
            if supervision is None:
                continue
            # Case 1: Action mismatch -- record False for all attributes.
            if not action_match:
                for key in supervision.keys():
                    if key in IGNORE_ATTRIBUTES:
                        continue
                    matches["attributes"].append(False)
            # Case 2: Action matches -- use model predictions for attributes.
            else:
                for key in supervision.keys():
                    if key in IGNORE_ATTRIBUTES:
                        continue
                    gt_key_vals = supervision[key]
                    model_key_vals = round_datum["attributes"][key]
                    if not len(gt_key_vals):
                        continue
                    # For fashion, this is a list -- multi label prediction.
                    if isinstance(gt_key_vals, list):
                        assert isinstance(model_key_vals, list), (
                            "Model should also predict a list for attributes"
                        )
    
                        recall = np.mean(
                            [(ii in model_key_vals) for ii in gt_key_vals]
                        )
                        if len(model_key_vals):
                            precision = np.mean(
                                [(ii in gt_key_vals) for ii in model_key_vals]
                            )
                        else:
                            precision = 0.
                        f1_score = (2 * recall * precision) / (recall + precision + 1e-5)
                        matches["attributes"].append(f1_score)
                    else:
                        # For furniture, this is a string -- single label prediction.
                        matches["attributes"].append(gt_key_vals == model_key_vals)

    # Compute the confusion matrix.
    all_actions = sorted(
        set(confusion_dict.keys()).union(
            {jj for ii in confusion_dict.values() for jj in ii}
        )
    )
    matrix = np.zeros((len(all_actions), len(all_actions)))
    for index, action in enumerate(all_actions):
        labels, counts = np.unique(confusion_dict[action], return_counts=True)
        for label, count in zip(labels, counts):
            matrix[all_actions.index(label), index] += count
    print( "action_accuracy", np.mean(matches["action"]))
    print("attribute_accuracy", np.mean(matches["attributes"]))
    return {
        "action_accuracy": np.mean(matches["action"]),
        # "action_perplexity": np.exp(-1 * np.mean(matches["perplexity"])),
        "attribute_accuracy": np.mean(matches["attributes"]),
        "confusion_matrix": matrix
    }


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,
                        help='json path for predicted') 
    parser.add_argument('--domain',
                        help='domain (furniture or fashion) ', default="furniture")
    args = parser.parse_args()
    input_path = args.input_path
    domain = args.domain 
    if domain == "furniture" :
        action_json_path = "furniture_devtest_dials_api_calls.json"
    else :
        action_json_path = "fashion_devtest_dials_api_calls.json"
    print("Reading: {}".format(action_json_path))
    with open(action_json_path, "r") as file_id:
        gt_actions = json.load(file_id)

    model_output_path = input_path 
    print("Reading: {}".format(model_output_path))
    with open(model_output_path, "r") as file_id:
        model_actions = json.load(file_id)
    action_metrics = evaluate_action_prediction(gt_actions, model_actions)
    print(action_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,
                        help='json path for predicted') 
    parser.add_argument('--domain',
                        help='domain (furniture or fashion) ', default="furniture")
    args = parser.parse_args()
    input_path = args.input_path
    domain = args.domain 
    if domain == "fashion" :
        action_json_path = "fashion_devtest_dials_api_calls.json"
    else :
        action_json_path = "furniture_devtest_dials_api_calls.json"
    print(action_json_path)
    print("Reading: {}".format(action_json_path))
    with open(action_json_path, "r") as file_id:
        gt_actions = json.load(file_id)

    model_output_path = input_path 
    print("Reading: {}".format(model_output_path))
    with open(model_output_path, "r") as file_id:
        model_actions = json.load(file_id)
    action_metrics = evaluate_action_prediction2(gt_actions, model_actions,False)
    print(action_metrics)