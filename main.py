import chz
from opcd.experiments.delta_on_policy import run as delta_on_policy_run
from opcd.experiments.off_policy import run as off_policy_run
from opcd.experiments.on_policy import run as on_policy_run
from opcd.experiments.off_policy import OffPolicyConfig, evaluate

def main(config: OffPolicyConfig):
    match config.experiment:
        case "delta-on-policy":
            delta_on_policy_run(config)
        case "off-policy":
            off_policy_run(config)
        case "on-policy":
            on_policy_run(config)
        case _:
            raise ValueError(f"Invalid experiment: {config.experiment}")

if __name__ == "__main__":
    chz.nested_entrypoint(main)
# RUN_PATH = "tinker://cc763284-c5fb-5208-8f27-94ef27b34aa0:train:0/sampler_weights/off-policy-k50-rank32-final"
# config = OffPolicyConfig()
# evaluate(RUN_PATH, config)
