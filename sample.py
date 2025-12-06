from opcd.experiments.off_policy import OffPolicyConfig, evaluate

RUN_PATH = "tinker://a7b8dfec-8675-56ca-b86c-6a27f179709a:train:0/sampler_weights/context-aware-on-policy-k50-rank32-10"
config = OffPolicyConfig()
evaluate(RUN_PATH, config)