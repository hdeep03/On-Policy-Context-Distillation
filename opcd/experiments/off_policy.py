import os
from typing import Optional

from opcd.dataset.hendrycks_math import train_dataset, test_dataset, is_correct

import tinker
from tinker import types
from tinker_cookbook.renderers import Renderer, get_renderer, TrainOnWhat
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.utils.format_colorized import format_colorized
from tinker_cookbook.renderers import Message
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier
from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.utils.ml_log import setup_logging
from rich.progress import track
import logging
import chz

logger = logging.getLogger(__name__)

@chz.chz
class OffPolicyConfig:
    teacher_model: str = "Qwen/Qwen3-8B"
    student_model: str = "Qwen/Qwen3-8B"
    seed: int = 42
    k: int = 20
    rank: int = 32 
    max_gen_tokens: int = 2000
    temperature: float = 0.0
    batch_size: int = 16 
    learning_rate: float = get_lr("Qwen/Qwen3-8B")

def create_teacher_prompt(renderer: Renderer, sample: dict, seed: int, k: int, prefill: Optional[str] = "Solution: ") -> types.ModelInput:
    system_message = [
        Message(role="system", content="You are a helpful assistant."),
    ]
    in_context_examples = train_dataset.shuffle(seed).select(range(k))
    in_context_examples_messages = []
    for example in in_context_examples:
        in_context_examples_messages.extend(
            [
                Message(role="user", content=f"Problem: {example['problem']}"),
                Message(role="assistant", content=f"Solution: {example['solution']}"),
            ]
        )
    user_message = [
        Message(role="user", content=f"Problem: {sample['problem']}"),
    ]
    chat_messages = system_message + in_context_examples_messages + user_message
    return renderer.build_generation_prompt(chat_messages, prefill=prefill)

def create_student_prompt(renderer: Renderer, sample: dict) -> types.ModelInput:
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content=f"Problem: {sample['problem']}"),
    ]
    return renderer.build_generation_prompt(messages)

def create_supervised_example(renderer: Renderer, sample: dict, teacher_response: str, max_length: int) -> types.Datum:
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content=f"Problem: {sample['problem']}"),
        Message(role="assistant", content=f"Solution: {teacher_response}"),
    ]
    return conversation_to_datum(messages, renderer, max_length=max_length, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
    

def get_renderer_name(model_name: str) -> str:
    match model_name:
        case "Qwen/Qwen3-8B":
            return "qwen3"
        case _:
            raise ValueError(f"Unknown model: {model_name}")

def run(config: OffPolicyConfig):
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=config.teacher_model)
    training_client = service_client.create_lora_training_client(
        base_model=config.student_model,
        rank=config.rank,
        seed=config.seed
    )
    experiment_name = f"off-policy-k{config.k}-rank{config.rank}"
    log_dir = f"./logs/off-policy/{config.k}-{config.rank}"
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
    n_batches = len(train_dataset) // config.batch_size
    total_steps = n_batches 

    # Generate teacher responses for the training corpus
    samples = []
    teacher_futures = []
    for sample in train_dataset_shuffled:
        prompt = create_teacher_prompt(teacher_renderer, sample, config.seed, config.k)
        sampling_params = types.SamplingParams(max_tokens=config.max_gen_tokens, temperature=config.temperature, stop=teacher_renderer.get_stop_sequences())
        future = sampling_client.sample(prompt=prompt, sampling_params=sampling_params, num_samples=1)
        teacher_futures.append(future)
        samples.append(sample)

    results = []
    for future in track(teacher_futures, description="Generating teacher responses..."):
        results.append(future.result())

    teacher_responses = [teacher_tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=False) for result in results]
    logger.info(f"Generated {len(teacher_responses)} teacher responses")

    for batch_idx in range(n_batches):
        lr_multiplier = compute_schedule_lr_multiplier("linear", batch_idx, total_steps)
        batch_slice = slice(batch_idx * config.batch_size, (batch_idx + 1) * config.batch_size)
        batch = samples[batch_slice]
        batch_teacher_responses = teacher_responses[batch_slice]
        supervised_examples = [create_supervised_example(student_renderer, sample, teacher_response, config.max_gen_tokens) for sample, teacher_response in zip(batch, batch_teacher_responses)]
        fwd_bwd_future = training_client.forward_backward(supervised_examples, loss_fn="cross_entropy")
        adam_params = types.AdamParams(learning_rate=config.learning_rate * lr_multiplier)
        optim_step_future = training_client.optim_step(adam_params)
        fwd_bwd_result = fwd_bwd_future.result()
        _ = optim_step_future.result()
        ml_logger.log_metrics(fwd_bwd_result.metrics, step=batch_idx)
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


def evaluate(sampler_path: str, config: OffPolicyConfig):
    service_client = tinker.ServiceClient()
    logger.info(f"Evaluating with sampler path: {sampler_path}")
    sampling_client = service_client.create_sampling_client(model_path=sampler_path)
    logger.info(f"Created sampling client")
    student_tokenizer = get_tokenizer(config.student_model)
    student_renderer = get_renderer(get_renderer_name(config.student_model), student_tokenizer)
    logger.info(f"Created student renderer")

    futures = []
    samples = []
    correct_count = 0
    for sample in test_dataset:
        prompt = create_student_prompt(student_renderer, sample)
        sampling_params = types.SamplingParams(max_tokens=config.max_gen_tokens, temperature=config.temperature, stop=student_renderer.get_stop_sequences())
        future = sampling_client.sample(prompt=prompt, sampling_params=sampling_params, num_samples=1)
        futures.append(future)
        samples.append(sample)
    
    results = []
    for future in track(futures, description="Evaluating..."):
        results.append(future.result())

    for result, sample in zip(results, samples):
        student_response = student_tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=False)
        if is_correct(sample, student_response):
            correct_count += 1

    print(f"Accuracy: {correct_count / len(samples)}; {correct_count} / {len(samples)}")

