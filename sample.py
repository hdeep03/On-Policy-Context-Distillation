from opcd.experiments.off_policy import OffPolicyConfig, evaluate

RUN_PATH = "tinker://5d49697d-4ebe-54ca-a5eb-6b3f8069fbe2:train:0/sampler_weights/delta-on-policy-k20-rank32-340"
config = OffPolicyConfig()
print(RUN_PATH)
evaluate(RUN_PATH, config)