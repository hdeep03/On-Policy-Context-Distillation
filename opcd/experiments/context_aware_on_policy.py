import logging
import os
from typing import List, Optional

import chz
from rich.progress import track
import tinker
from tinker import types
import numpy as np
from tinker_cookbook.renderers import Renderer, get_renderer, TrainOnWhat
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.renderers import Message
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier
from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.utils.ml_log import setup_logging

from opcd.dataset.hendrycks_math import train_dataset, test_dataset, is_correct
from opcd.experiments.off_policy import OffPolicyConfig, get_renderer_name, create_teacher_prompt, create_student_prompt

logger = logging.getLogger(__name__)

def run(config: OffPolicyConfig):
    service_client = tinker.ServiceClient()
    teacher_sampler = service_client.create_sampling_client(base_model=config.teacher_model)
    training_client = service_client.create_lora_training_client(
        base_model=config.student_model,
        rank=config.rank,
        seed=config.seed
    )
    experiment_name = f"context-aware-on-policy-k{config.k}-rank{config.rank}"
    log_dir = f"./logs/context-aware-on-policy/{config.k}-{config.rank}"
    os.makedirs(log_dir, exist_ok=True)
    checkpoints_path = os.path.join(log_dir, "checkpoints.jsonl")
    if not os.path.exists(checkpoints_path):
        open(checkpoints_path, "w").close()
    ml_logger = setup_logging(log_dir=log_dir, wandb_project="context-distillation", wandb_name=experiment_name, config=config)
    ml_logger.log_hparams(config)
    student_tokenizer = get_tokenizer(config.student_model)
    student_renderer = get_renderer(get_renderer_name(config.student_model), student_tokenizer)
    teacher_tokenizer = get_tokenizer(config.teacher_model)
    teacher_renderer = get_renderer(get_renderer_name(config.teacher_model), teacher_tokenizer)

    train_dataset_shuffled = train_dataset.shuffle(config.seed)
    n_batches = len(train_dataset_shuffled) // config.batch_size
    total_steps = n_batches

    sampling_params = types.SamplingParams(max_tokens=config.max_gen_tokens, temperature=config.temperature, stop=student_renderer.get_stop_sequences())
    teacher_futures = []
    for batch_idx in range(n_batches):
        student_sampler = training_client.save_weights_and_get_sampling_client()
        lr_multiplier = compute_schedule_lr_multiplier("linear", batch_idx, total_steps)
        batch = train_dataset_shuffled.select(range(batch_idx * config.batch_size, (batch_idx + 1) * config.batch_size))
        student_prompts = [create_student_prompt(student_renderer, sample) for sample in batch]
        teacher_prompts = [create_teacher_prompt(teacher_renderer, sample, config.seed, config.k, prefill=None) for sample in batch]
        short_teacher_prompts = [create_teacher_prompt(teacher_renderer, sample, config.seed, 0, prefill=None) for sample in batch]

        student_futures = []
        for student_prompt in student_prompts:
            future = student_sampler.sample(prompt=student_prompt, sampling_params=sampling_params, num_samples=1)
            student_futures.append(future)
        student_responses: List[types.SampleResponse] = [student_future.result() for student_future in student_futures]

        teacher_futures = []
        short_teacher_futures = []
        for teacher_prompt, short_teacher_prompt, student_response in zip(teacher_prompts, short_teacher_prompts, student_responses):
            teacher_prompt = types.ModelInput.from_ints(teacher_prompt.to_ints() + student_response.sequences[0].tokens)
            short_teacher_prompt = types.ModelInput.from_ints(short_teacher_prompt.to_ints() + student_response.sequences[0].tokens)
            future = teacher_sampler.compute_logprobs(teacher_prompt)
            teacher_futures.append(future)
            short_teacher_future = teacher_sampler.compute_logprobs(short_teacher_prompt)
            short_teacher_futures.append(short_teacher_future)

        teacher_logprobs: List[List[Optional[float]]] = [teacher_future.result() for teacher_future in teacher_futures]
        short_teacher_logprobs: List[List[Optional[float]]] = [short_teacher_future.result() for short_teacher_future in short_teacher_futures]
        data = []
        mean_advantages = []
        mean_deltas = []
        mean_weights = []
        mean_advantage_weights = []
        for teacher_logprob, short_teacher_logprob, student_response, student_prompt in zip(teacher_logprobs, short_teacher_logprobs, student_responses, student_prompts):
            student_logprobs = student_response.sequences[0].logprobs
            n = len(student_logprobs)
            delta = np.array([teacher_logp - short_teacher_logp for teacher_logp, short_teacher_logp in zip(teacher_logprob[-n:], short_teacher_logprob[-n:])])
            context_relevance = np.clip(np.abs(delta), 0, 2.0)
            sensitivity = 2.0
            weights = 1.0 + (sensitivity * context_relevance)
            advantage = np.array([teacher_logp - student_logp for teacher_logp, student_logp in zip(teacher_logprob[-n:], student_logprobs)])
            weighted_advantage = advantage * weights
            prompt_tokens = student_prompt.to_ints()
            mean_advantages.append(advantage.mean())
            mean_deltas.append(delta.mean())
            mean_weights.append(weights.mean())
            mean_advantage_weights.append(weighted_advantage.mean())
            tokens = prompt_tokens + student_response.sequences[0].tokens
            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]
            all_logprobs = [0.0] * (len(prompt_tokens) - 1) + student_logprobs
            all_advantages = [0.0] * (len(prompt_tokens) - 1) + weighted_advantage.tolist()
            datum = types.Datum(
                model_input=types.ModelInput.from_ints(input_tokens),
                loss_fn_inputs={
                    "target_tokens": target_tokens,
                    "logprobs": all_logprobs,
                    "advantages": all_advantages
                }
            )
            data.append(datum)
        
        fwd_bwd_future = training_client.forward_backward(data, loss_fn="importance_sampling")
        optim_params = types.AdamParams(learning_rate=config.learning_rate * lr_multiplier)
        optim_step_future = training_client.optim_step(optim_params)
        fwd_bwd_result = fwd_bwd_future.result()
        _ = optim_step_future.result()
        metrics = {
            **fwd_bwd_result.metrics,
            "mean_advantages": np.mean(mean_advantages),
            "mean_deltas": np.mean(mean_deltas),
            "mean_weights": np.mean(mean_weights),
            "mean_advantage_weights": np.mean(mean_advantage_weights),
        }
        ml_logger.log_metrics(metrics, step=batch_idx)
        if batch_idx % 10 == 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{experiment_name}-{batch_idx}",
                log_path=log_dir,
                loop_state={
                    "batch_idx": batch_idx,
                },
                kind="both"
            )
    
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=f"{experiment_name}-final",
        log_path=log_dir,
        loop_state={
            "batch_idx": n_batches,
        },
        kind="both"
    )