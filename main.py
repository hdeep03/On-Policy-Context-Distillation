import chz
from opcd.experiments.context_aware_on_policy import run
from opcd.experiments.off_policy import OffPolicyConfig, evaluate

if __name__ == "__main__":
    chz.nested_entrypoint(run)
# RUN_PATH = "tinker://cc763284-c5fb-5208-8f27-94ef27b34aa0:train:0/sampler_weights/off-policy-k50-rank32-final"
# config = OffPolicyConfig()
# evaluate(RUN_PATH, config)
