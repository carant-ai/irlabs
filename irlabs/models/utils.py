from typing import Dict

def combine_dict(*args: Dict):
    new_dict = {}
    for d in args:
        for key, value in d.items():
            if new_dict.get(key):
                continue

            new_dict[key] = value
    return new_dict


def resolve_config():
