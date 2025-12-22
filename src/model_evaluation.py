# import sys

# sys.path.append(".")
# config = get_linemod_config("Linemod_preprocessed_small")

# points, diameters = config.load_all_models_3d('mm')

# print("Loaded model diameters:", diameters)

from utils.linemod_config import get_linemod_config
from metrics.ADD_metric import compute_ADD_metric_quaternion, compute_ADDs_metric_quaternion
import pandas as pd



from collections import defaultdict
import torch
from tqdm import tqdm

def evaluate_comprehensive(model, dataloader, device, gt_function, inference_function, model_points: dict, model_diameters: dict):
    model.eval()

    # --- MAPPING ID TO NAMES (LineMOD Standard) ---
    id_to_name = {
        1: 'ape', 2: 'benchvise', 4: 'camera', 5: 'can', 6: 'cat',
        8: 'driller', 9: 'duck', 10: 'eggbox', 11: 'glue',
        12: 'holepuncher', 13: 'iron', 14: 'lamp', 15: 'phone'
    }

    all_object_ids = list(id_to_name.keys())

    errors_dict = defaultdict(list)
    accuracy_stats = defaultdict(lambda: {'correct_add': 0, 'correct_adds': 0, 'total': 0})

    print("\n🚀 Starting Comprehensive Benchmark (ADD Error + ADD-0.1d Accuracy)...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            gt_rot, gt_trans = gt_function(batch, device)
            pred_rot, pred_trans = inference_function(model, batch, device)
            object_ids = batch['object_id']

            batch_size = gt_rot.shape[0]

            for i in range(batch_size):
                curr_id = int(object_ids[i])
                if curr_id not in all_object_ids:
                    continue
                model_pts = model_points[curr_id]
                gt_q = gt_rot[i].cpu().numpy()
                gt_t = gt_trans[i].cpu().numpy()
                pred_q = pred_rot[i].cpu().numpy()
                pred_t = pred_trans[i].cpu().numpy()

                diameter = model_diameters[curr_id]

                add_error = compute_ADD_metric_quaternion(model_pts, pred_q, pred_t, gt_q, gt_t)
                adds_error = compute_ADDs_metric_quaternion(model_pts, pred_q, pred_t, gt_q, gt_t)
                add_rot_error = compute_ADD_metric_quaternion(model_pts, pred_q, gt_t, gt_q, gt_t)
                adds_rot_error = compute_ADDs_metric_quaternion(model_pts, pred_q, gt_t, gt_q, gt_t)

                correct_add = add_error < 0.1 * diameter
                correct_adds = adds_error < 0.1 * diameter
                accuracy_stats[curr_id]['correct_add'] += int(correct_add)
                accuracy_stats[curr_id]['correct_adds'] += int(correct_adds)
                accuracy_stats[curr_id]['total'] += 1


                errors_dict[curr_id].append({
                    'ADD': add_error,
                    'ADD-S': adds_error,
                    'ADD-Rot': add_rot_error,
                    'ADD-S-Rot': adds_rot_error
                })

    return errors_dict, accuracy_stats

            
def generate_dataframe(errors_dict: dict, accuracy_stats: dict, all_object_ids: dict, object_diameters: dict, um: str = 'mm') -> pd.DataFrame:
    report_rows = []

    for obj_id, errors in errors_dict.items():
        df = pd.DataFrame(errors)
        mean_add = df['ADD'].mean()
        mean_adds = df['ADD-S'].mean()
        mean_add_rot = df['ADD-Rot'].mean()
        mean_adds_rot = df['ADD-S-Rot'].mean()

        acc_stats = accuracy_stats[obj_id]
        add_accuracy = acc_stats['correct_add'] / acc_stats['total'] if acc_stats['total'] > 0 else 0.0
        adds_accuracy = acc_stats['correct_adds'] / acc_stats['total'] if acc_stats['total'] > 0 else 0.0

        report_rows.append({
            'Object ID': obj_id,
            'Object Name': all_object_ids[obj_id],
            'Diameter ({})'.format(um): object_diameters[obj_id],
            f'Mean ADD ({um})': mean_add,
            f'Mean ADD-S ({um})': mean_adds,
            f'Mean ADD-Rot ({um})': mean_add_rot,
            f'Mean ADD-S-Rot ({um})': mean_adds_rot,
            'ADD-0.1d Accuracy (%)': add_accuracy * 100,
            'ADD-S-0.1d Accuracy (%)': adds_accuracy * 100
        })

    report_df = pd.DataFrame(report_rows)

    # Overall statistics
    overall_add = report_df[f'Mean ADD ({um})'].mean()
    overall_adds = report_df[f'Mean ADD-S ({um})'].mean()
    overall_add_rot = report_df[f'Mean ADD-Rot ({um})'].mean()
    overall_adds_rot = report_df[f'Mean ADD-S-Rot ({um})'].mean()
    overall_add_accuracy = report_df['ADD-0.1d Accuracy (%)'].mean()
    overall_adds_accuracy = report_df['ADD-S-0.1d Accuracy (%)'].mean()
    overall_row = {
        'Object ID': 'Overall',
        'Object Name': 'All Objects',
        'Diameter ({})'.format(um): '-',
        f'Mean ADD ({um})': overall_add,
        f'Mean ADD-S ({um})': overall_adds,
        f'Mean ADD-Rot ({um})': overall_add_rot,
        f'Mean ADD-S-Rot ({um})': overall_adds_rot,
        'ADD-0.1d Accuracy (%)': overall_add_accuracy,
        'ADD-S-0.1d Accuracy (%)': overall_adds_accuracy
    }

    return report_df


def save_report_to_csv(report_df: pd.DataFrame, file_path: str):
    report_df.to_csv(file_path, index=False)
    print(f"\n📊 Evaluation report saved to {file_path}")

def evalutation_pipeline(model, dataloader, device, gt_function, inference_function, model_points: dict, model_diameters: dict, report_file_path: str, um: str = 'mm') -> pd.DataFrame:
    errors_dict, accuracy_stats = evaluate_comprehensive(
        model, dataloader, device, gt_function, inference_function, model_points, model_diameters
    )

    id_to_name = {
        1: 'ape', 2: 'benchvise', 4: 'camera', 5: 'can', 6: 'cat',
        8: 'driller', 9: 'duck', 10: 'eggbox', 11: 'glue',
        12: 'holepuncher', 13: 'iron', 14: 'lamp', 15: 'phone'
    }

    report_df = generate_dataframe(errors_dict, accuracy_stats, id_to_name, model_diameters, um)

    save_report_to_csv(report_df, report_file_path)

    return report_df




