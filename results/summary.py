import os
import json
import math

DATASETS = ['bace', 'bbbp', 'hiv', 'tox21']
MODELS = [
    'GAEKAN_trained_encoder_200k',
    'GAEKAN_trained_encoder_500k',
    'GAEKAN_trained_encoder_1m',
    'GAEKAN_modified_trained_encoder_200k',
    'GAEKAN_modified_trained_encoder_500k',
    'GAEKAN_modified_trained_encoder_1m',
    'GAEKAN_mini_trained_encoder_200k',
    'GAEKAN_mini_trained_encoder_500k',
    'GAEKAN_mini_trained_encoder_1m',
    'GAEMLP_trained_encoder_200k',
    'GAEMLP_trained_encoder_500k',
    'GAEMLP_trained_encoder_1m',
]

# Files produced by KAN_datasets_only_testing.py: end-to-end KAN, trained
# from scratch per dataset (no separate pretrained encoder).
ONLY_KAN_FILES = {
    'bace':  ('bace_only_KAN.json',  'bace_results'),
    'bbbp':  ('bbbp_only_KAN.json',  'bbbp_results'),
    'hiv':   ('hiv_only_KAN.json',   'hiv_results'),
    'tox21': ('tox21_only_KAN.json', 'tox21_results'),
}

def load_results():
    results = {}
    for model in MODELS:
        for dataset in DATASETS:
            fname = f'{dataset}{model}.json'
            if not os.path.isfile(fname):
                continue
            with open(fname) as f:
                data = json.load(f)
            key = f'{dataset}{model}'
            aucs = data.get(key, [])
            if aucs:
                results[(dataset, model)] = aucs
    return results

def load_only_kan_results():
    results = {}
    for dataset, (fname, key) in ONLY_KAN_FILES.items():
        if not os.path.isfile(fname):
            continue
        with open(fname) as f:
            data = json.load(f)
        aucs = data.get(key, [])
        if aucs:
            results[dataset] = aucs
    return results

def summarise(aucs):
    n    = len(aucs)
    mean = sum(aucs) / n
    std  = math.sqrt(sum((x - mean) ** 2 for x in aucs) / n)
    return {
        'n':            n,
        'mean':         mean,
        'std':          std,
        'min':          min(aucs),
        'max':          max(aucs),
        'below_chance': sum(1 for x in aucs if x < 0.5),
    }

def print_header_row(col_w, stat_w):
    header = (f"{'Model / Dataset':<{col_w}}"
              f"{'Mean':>{stat_w}}"
              f"{'Std':>{stat_w}}"
              f"{'Min':>{stat_w}}"
              f"{'Max':>{stat_w}}"
              f"{'<0.5':>{stat_w}}"
              f"{'n':>{stat_w}}")
    print(header)
    print('-' * 72)

def print_table(results):
    col_w  = 28
    stat_w = 8

    arch_groups = [
        ('KAN',          [m for m in MODELS if m.startswith('GAEKAN_trained')]),
        ('KAN MODIFIED', [m for m in MODELS if m.startswith('GAEKAN_modified')]),
        ('KAN MINI',     [m for m in MODELS if m.startswith('GAEKAN_mini')]),
        ('MLP',          [m for m in MODELS if 'MLP' in m]),
    ]

    for arch_name, arch_models in arch_groups:
        print(f"\n{'='*72}")
        print(f"  {arch_name} ENCODER RESULTS")
        print(f"{'='*72}")
        print_header_row(col_w, stat_w)

        for model in arch_models:
            short = (model
                     .replace('GAEKAN_modified_trained_encoder_', '')
                     .replace('GAEKAN_mini_trained_encoder_', '')
                     .replace('GAEKAN_trained_encoder_', '')
                     .replace('GAEMLP_trained_encoder_', ''))
            print(f"\n  {arch_name} {short.upper()}")
            for dataset in DATASETS:
                if (dataset, model) not in results:
                    print(f"  {dataset.upper():<{col_w-2}}  {'(not found)':>{stat_w*5}}")
                    continue
                s = summarise(results[(dataset, model)])
                row = (f"  {dataset.upper():<{col_w-2}}"
                       f"{s['mean']:>{stat_w}.4f}"
                       f"{s['std']:>{stat_w}.4f}"
                       f"{s['min']:>{stat_w}.4f}"
                       f"{s['max']:>{stat_w}.4f}"
                       f"{s['below_chance']:>{stat_w}}"
                       f"{s['n']:>{stat_w}}")
                print(row)

    # Cross-model comparison: best mean per dataset
    print(f"\n{'='*72}")
    print(f"  BEST MEAN AUC PER DATASET")
    print(f"{'='*72}")
    print(f"{'Dataset':<10} {'Best Model':<{col_w+8}} {'Mean AUC':>10}")
    print('-' * 50)
    for dataset in DATASETS:
        best_auc, best_model = -1, None
        for model in MODELS:
            if (dataset, model) in results:
                aucs = results[(dataset, model)]
                m = sum(aucs) / len(aucs)
                if m > best_auc:
                    best_auc, best_model = m, model
        if best_model:
            print(f"{dataset.upper():<10} {best_model:<{col_w+8}} {best_auc:>10.4f}")
        else:
            print(f"{dataset.upper():<10} {'(no results found)':<{col_w+8}}")


def print_only_kan_table(only_kan_results):
    col_w  = 28
    stat_w = 8

    print(f"\n{'='*72}")
    print(f"  KAN (END-TO-END, NO PRETRAINED ENCODER) RESULTS")
    print(f"{'='*72}")
    print_header_row(col_w, stat_w)
    print(f"\n  KAN ONLY")
    for dataset in DATASETS:
        if dataset not in only_kan_results:
            print(f"  {dataset.upper():<{col_w-2}}  {'(not found)':>{stat_w*5}}")
            continue
        s = summarise(only_kan_results[dataset])
        row = (f"  {dataset.upper():<{col_w-2}}"
               f"{s['mean']:>{stat_w}.4f}"
               f"{s['std']:>{stat_w}.4f}"
               f"{s['min']:>{stat_w}.4f}"
               f"{s['max']:>{stat_w}.4f}"
               f"{s['below_chance']:>{stat_w}}"
               f"{s['n']:>{stat_w}}")
        print(row)


if __name__ == '__main__':
    results = load_results()
    only_kan_results = load_only_kan_results()

    if not results and not only_kan_results:
        print("No result JSON files found in the current directory.")
    else:
        if results:
            print_table(results)
        print_only_kan_table(only_kan_results)
        print()
