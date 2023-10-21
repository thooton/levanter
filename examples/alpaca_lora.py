# Modified version of [alpaca.py] to use LoRA instead of full finetuning.

import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import jax.random as jrandom
import transformers
import wandb

import haliax as hax

import levanter
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.lora import (
    LoraConfig,
    lora_trainable_params_filter,
    loraize,
    save_merged_hf_checkpoint_callback,
    save_peft_checkpoint_callback,
)
from levanter.models.lm_model import LmExample, LmHeadModel
from levanter.trainer import Trainer
from levanter.utils.jax_utils import parameter_count
from levanter.utils.py_utils import non_caching_cycle


# This is a bit of a hack to make sure we can load the module no matter where we run it from.
# You can also just set the PYTHONPATH environment variable.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # noqa: E402

import alpaca  # noqa: E402
from alpaca import SupervisedDataset  # noqa: E402


logger = logging.getLogger(__name__)


@dataclass
class TrainArgs(alpaca.TrainArgs):
    lora: LoraConfig = LoraConfig()

    # should we save merged (i.e. not peft) checkpoints?
    merged_hf_save_path: Optional[str] = None  # path to save merged hf checkpoints
    merged_hf_upload: Optional[str] = None


def train(config: TrainArgs):
    config.trainer.initialize(config)

    # Since Levanter has different implementations of models from HF, we need to convert the HF checkpoint.
    # This class is a wrapper around the HF checkpoint converter that also downloads the checkpoint if necessary.
    converter = HFCheckpointConverter.from_hf(config.model_name_or_path, trust_remote_code=config.trust_remote_code)
    model_config = converter.default_config

    # Randomness in JAX is tightly controlled. We pass around a key that is used to generate random numbers.
    training_key, lora_key = jrandom.split(jrandom.PRNGKey(config.trainer.seed), 2)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        cache_dir=config.model_cache_dir,
        model_max_length=model_config.Pos.size,
        padding_side="right",
    )
    num_new_tokens = alpaca.add_special_tokens(tokenizer)

    # modify converter to use our tokenizer, mostly so it saves the right vocab
    converter = converter.replaced(tokenizer=tokenizer)

    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    with config.trainer.device_mesh:
        # how we shard parameters across devices
        parameter_axis_mapping = config.trainer.parameter_axis_mapping

        # load the underlying hf model
        logger.info(f"Loading pretrained model from {converter.reference_checkpoint}")
        model: LmHeadModel = converter.load_pretrained(model_config, axis_mapping=parameter_axis_mapping)

        logger.info(f"Added {num_new_tokens} new tokens")
        # this must be in jit b/c it uses arrays across accelerators (b/c of FSDP)
        model = hax.named_jit(lambda m: m.resize_vocab(len(tokenizer)))(model)

        # Major difference from Alpaca: we loraize the model.

        @hax.named_jit(axis_resources=parameter_axis_mapping, donate_args=(True))
        def loraize_hf_model(model):
            return loraize(model, config.lora, key=lora_key)

        model = loraize_hf_model(model)

        lora_param_filter = lora_trainable_params_filter(model)

        def compute_loss(model: LmHeadModel, example: LmExample, key=None):
            return model.compute_loss(example, key=key).scalar()

        trainer = Trainer(config.trainer, optimizer, compute_loss, is_trainable=lora_param_filter)

        # end major difference from Alpaca

        trainer.add_default_hooks()
        state = trainer.initial_state(training_key, model=model)

        # log some info about the model
        all_param_count = parameter_count(state.model)
        just_lora_params = parameter_count(trainer.trainable_params_only(state.model))

        wandb.summary["parameter_count"] = all_param_count
        wandb.summary["trainable_parameter_count"] = just_lora_params
        logger.info(f"Total parameter count: {all_param_count}")
        logger.info(f"Trainable parameter count: {just_lora_params}")
        logger.info(f"Fraction of parameters that are trainable: {just_lora_params * 1.0 / all_param_count%.3}")

        train_dataset = SupervisedDataset(config.data_cache_dir, model.Pos, model.KeyPos, config.data, tokenizer)
        # Levanter has two kinds of data loaders: sharded and replicated. Replicated is simpler and allows for
        # single pass training. Sharded only loads a subset of the data on each device, and is more efficient for large
        # datasets. We use replicated here since the dataset is small.
        loader = trainer.replicated_loader(train_dataset, trainer.TrainBatch)
        loader = non_caching_cycle(loader)

        if state.step != 0:
            logger.info(f"Resuming training from step {state.step}")
            for i in range(state.step):
                next(loader)  # type: ignore

        # Save HF PEFT checkpoints periodically (and at the end of training), which is just the lora weights
        if config.hf_save_path is not None:
            full_save_path = os.path.join(config.hf_save_path, trainer.run_id)
            trainer.add_hook(
                save_peft_checkpoint_callback(
                    full_save_path, config.lora, config.model_name_or_path, tokenizer, config.hf_upload
                ),
                every=config.hf_save_steps,
            )

        # Save merged HF checkpoints if requested
        if config.merged_hf_save_path is not None:
            full_save_path = os.path.join(config.merged_hf_save_path, trainer.run_id)
            trainer.add_hook(
                save_merged_hf_checkpoint_callback(full_save_path, converter, config.merged_hf_upload),
                every=config.hf_save_steps,
            )

        trainer.train(state, loader)


if __name__ == "__main__":
    levanter.config.main(train)()
