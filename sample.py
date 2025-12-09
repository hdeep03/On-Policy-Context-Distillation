import chz
from opcd.experiments.off_policy import OffPolicyConfig, evaluate_teacher

if __name__ == "__main__":
    print(chz.nested_entrypoint(evaluate_teacher))