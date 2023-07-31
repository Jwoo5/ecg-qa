import logging
import argparse
import os
import re
import pprint
import time

import wandb
import openai

from fairseq_signals.dataclass.initialize import add_defaults, hydra_init
from fairseq_signals.dataclass.utils import omegaconf_no_object_check
from fairseq_cli.validate import main as pre_main
from fairseq_signals.logging import metrics, progress_bar
from fairseq_signals.dataclass.configs import Config
from fairseq_signals.utils.utils import reset_logging

from fairseq_signals import distributed_utils, tasks
from fairseq_signals.utils import checkpoint_utils, options, utils

import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from omegaconf import OmegaConf, open_dict, DictConfig

logger = logging.getLogger("fairseq_cli.llm_expr")

def main(cfg: DictConfig, override_args=None):
    torch.multiprocessing.set_sharing_strategy("file_system")

    utils.import_user_module(cfg.common)

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)
    
    if cfg.distributed_training.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    if override_args is not None:
        overrides = vars(override_args)
    else:
        overrides = {}

    overrides.update({"task": {"data": cfg.task.data}})
    model_overrides = eval(getattr(cfg.common_eval, "model_overrides", "{}"))
    overrides.update(model_overrides)

    # Load model
    logger.info(f"loading model from {cfg.common_eval.path}")
    model, saved_cfg, task = checkpoint_utils.load_model_and_task(
        cfg.common_eval.path,
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix
    )
    
    task = tasks.setup_task(cfg.task)

    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False) and p.requires_grad)
        )
    )

    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False) and p.requires_grad)
        )
    )

    # Move model to GPU
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Print args
    logger.info(pprint.pformat(dict(saved_cfg)))

    # Build criterion
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.eval()

    def _fp_convert_sample(sample):
        def apply_half(t):
            if t.dtype in [torch.float64, torch.float32, torch.int16]:
                return t.to(dtype = torch.half)
            return t
            # return t.to(dtype = torch.half)
        
        def apply_float(t):
            if t.dtype in [torch.float64, torch.float32, torch.int16]:
                return t.to(dtype = torch.float)
            return t
        
        if use_fp16:
            sample = utils.apply_to_sample(apply_half, sample)
        else:
            sample = utils.apply_to_sample(apply_float, sample)
        
        return sample

    import pandas as pd
    grounding_classes = pd.read_csv(os.path.join("..", "..", "..", os.path.dirname(__file__), "grounding_classes.csv"))
    grounding_classes = dict(grounding_classes["class"])
    qa_classes = pd.read_csv(os.path.join("..", "..", "..", os.path.dirname(__file__), "qa_classes.csv"))
    qa_classes = dict(qa_classes["class"])
    leads = [
        "lead I", "lead II", "lead III", "lead aVR", "lead aVL", "lead aVF",
        "lead V1", "lead V2", "lead V3", "lead V4", "lead V5", "lead V6"
    ]
    lead_pattern = r"(lead (I|II|III|aVR|aVL|aVF|V1|V2|V3|V4|V5|V6))|((limb|chest) leads)"

    openai.api_key = cfg.openai_api_key
    if hasattr(cfg, "openai_organization"):
        openai.organization = cfg.openai_organization

    if (
        distributed_utils.is_master(cfg.distributed_training)
        and cfg.common.wandb_project is not None
        and cfg.common.wandb_entity is not None
    ):
        wandb.init(
            project=cfg.common.wandb_project,
            entity=cfg.common.wandb_entity,
            reinit=False,
            name=os.environ.get("WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir))
        )
        wandb.config.update(cfg)

    for subset in cfg.dataset.valid_subset.split(","):
        os.mkdir(subset)
        subset = subset.strip()
        task.load_dataset(subset, combine=False, epoch=1, task_cfg=cfg.task)
        dataset = task.dataset(subset)

        logger.info("begin validation on {} subset".format(subset))

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=cfg.dataset.max_tokens,
            max_signals=cfg.dataset.batch_size,
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=cfg.dataset.num_workers,
            data_buffer_size=cfg.dataset.data_buffer_size
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_file = cfg.common.log_file,
            log_interval=cfg.common.log_interval,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=None,
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=None,
            wandb_entity=None,
            wandb_run_name=None,
            azureml_logging=False
        )
        
        examplar_buffers = dict()
        total = {"question_type1": dict(), "question_type2": dict(), "question_type3": dict()}
        correct = {"question_type1": dict(), "question_type2": dict(), "question_type3": dict()}
        inner_total = {"question_type1": dict(), "question_type2": dict(), "question_type3": dict()}
        inner_correct = {"question_type1": dict(), "question_type2": dict(), "question_type3": dict()}
        num = 0
        for sample in progress:
            with torch.no_grad():
                sample = utils.move_to_cuda(sample) if use_cuda else sample
                sample = _fp_convert_sample(sample)
                for i in range(len(sample["id"])):
                    if sample["question_type1"][i].item() not in total["question_type1"]:
                        total["question_type1"][sample["question_type1"][i].item()] = 0
                        correct["question_type1"][sample["question_type1"][i].item()] = 0
                    if sample["question_type2"][i].item() not in total["question_type2"]:
                        total["question_type2"][sample["question_type2"][i].item()] = 0
                        correct["question_type2"][sample["question_type2"][i].item()] = 0
                    if sample["question_type3"][i].item() not in total["question_type3"]:
                        total["question_type3"][sample["question_type3"][i].item()] = 0
                        correct["question_type3"][sample["question_type3"][i].item()] = 0

                    if sample["question_type1"][i].item() not in inner_total["question_type1"]:
                        inner_total["question_type1"][sample["question_type1"][i].item()] = 0
                        inner_correct["question_type1"][sample["question_type1"][i].item()] = 0
                    if sample["question_type2"][i].item() not in inner_total["question_type2"]:
                        inner_total["question_type2"][sample["question_type2"][i].item()] = 0
                        inner_correct["question_type2"][sample["question_type2"][i].item()] = 0
                    if sample["question_type3"][i].item() not in inner_total["question_type3"]:
                        inner_total["question_type3"][sample["question_type3"][i].item()] = 0
                        inner_correct["question_type3"][sample["question_type3"][i].item()] = 0
                    
                    prompt = "These are the interpretations of each ECG along with their scores. "
                    prompt += "Higher score means more certainty about the interpretation.\n\n"
                    if sample["valid_classes"][i].tolist() == list(range(66, 78)):
                        for j in range(12):
                            prompt += f"Interpretation of the ECG in {leads[j]}:\n"
                            if f"{sample['ecg_id'][i][0]}_{j}" in examplar_buffers:
                                prompt += examplar_buffers[f"{sample['ecg_id'][i][0]}_{j}"]
                            else:
                                source = sample["net_input"]["ecg"][i]
                                mask = source.new_ones(source.shape).bool()
                                mask[j] = 0
                                source[mask] = 0
                                net_input = {"source": source[None, :, :]}
                                net_output = model(**net_input)
                                logits = model.get_logits(net_output).float()
                                scores = logits[0].sigmoid()
                                outputs = torch.where(scores > 0.5)[0]
                                statements = "\n".join([f"{grounding_classes[i.item()]}: {scores[i].item():.3f}" for i in outputs])
                                statements += "\n\n"
                                examplar_buffers[f"{sample['ecg_id'][i][0]}_{j}"] = statements
                                prompt += statements
                    elif (searched := re.search(lead_pattern, sample["question"][i])) is not None:
                        searched = searched.group()
                        if searched == "limb leads":
                            lead = [0, 1, 2, 3, 4, 5]
                            lead_name = searched
                        elif searched == "chest leads":
                            lead = [6, 7, 8, 9, 10, 11]
                            lead_name = searched
                        else:
                            lead = leads.index(searched)
                            lead_name = searched
                            searched = lead

                        prompt += f"Interpretation of the ECG in {lead_name}:\n"
                        if f"{sample['ecg_id'][i][0]}_{searched}" in examplar_buffers:
                            prompt += examplar_buffers[f"{sample['ecg_id'][i][0]}_{searched}"]
                        else:
                            source = sample["net_input"]["ecg"][i]
                            mask = source.new_ones(source.shape).bool()
                            mask[lead] = 0
                            source[mask] = 0
                            net_input = {"source": source[None, :, :]}
                            net_output = model(**net_input)
                            logits = model.get_logits(net_output).float()
                            scores = logits[0].sigmoid()
                            outputs = torch.where(scores > 0.5)[0]
                            statements = "\n".join([f"{grounding_classes[i.item()]}: {scores[i].item():.3f}" for i in outputs])
                            statements += "\n\n"
                            examplar_buffers[f"{sample['ecg_id'][i][0]}_{searched}"] = statements
                            prompt += statements
                    else:
                        # single
                        if len(sample["ecg_id"][i]) == 1:
                            prompt += "Interpretation of the ECG:\n"
                        else:
                            if "first ECG" in sample["question"][i]:
                                prompt += "Interpretation of the first ECG:\n"
                            else:
                                prompt += "Interpretation of the previous ECG:\n"

                        if sample["ecg_id"][i][0] in examplar_buffers:
                            prompt += examplar_buffers[sample["ecg_id"][i][0]]
                        else:
                            net_input = {
                                "source": sample["net_input"]["ecg"][i][None, :, :]
                            }
                            net_output = model(**net_input)
                            logits = model.get_logits(net_output).float()
                            scores = logits[0].sigmoid()
                            outputs = torch.where(scores > 0.5)[0]
                            statements = "\n".join([f"{grounding_classes[i.item()]}: {scores[i].item():.3f}" for i in outputs])
                            statements += "\n\n"
                            examplar_buffers[sample["ecg_id"][i][0]] = statements
                            prompt += statements
                            
                        # comparison
                        if len(sample["ecg_id"][i]) == 2:
                            if "second ECG" in sample["question"][i]:
                                prompt += "Interpretation of the second ECG:\n"
                            else:
                                prompt += "Interpretation of the recent ECG:\n"
                            if sample["ecg_id"][i][-1] in examplar_buffers:
                                prompt += examplar_buffers[sample["ecg_id"][i][-1]]
                            else:
                                net_input = {
                                    "source": sample["net_input"]["ecg_2"][i][None, :, :]
                                }
                                net_output = model(**net_input)
                                logits = model.get_logits(net_output).float()
                                scores = logits[0].sigmoid()
                                outputs = torch.where(scores > 0.5)[0]
                                statements = "\n".join([f"{grounding_classes[i.item()]}: {scores[i].item():.3f}" for i in outputs])
                                statements += "\n\n"
                                examplar_buffers[sample["ecg_id"][i][-1]] = statements
                                prompt += statements

                    prompt += "Question: "
                    prompt += sample["question"][i] + "\n"
                    prompt += "Options: "
                    prompt += ", ".join([qa_classes[c.item()] for c in sample["valid_classes"][i]])
                    # if not verify questions
                    if not sample["question_type2"][i].item() in [0, 3, 6]:
                        prompt += ", None"
                    prompt += "\n\n"
                    prompt += "Only answer based on the given Options without any explanation."

                    answer = set([qa_classes[i.item()].lower() for i in torch.where(sample["answer"][i])[0]])
                    if len(answer) == 0:
                        answer = {"none"}

                    while True:
                        try:
                            if cfg.openai_model == "gpt-4":
                                completion = openai.ChatCompletion.create(
                                    model="gpt-4",
                                    messages=[{"role": "user", "content": prompt}],
                                    temperature=0,
                                )
                                llm_answer = completion["choices"][0]["message"]["content"].lower()
                            elif cfg.openai_model == "gpt-3.5-turbo":
                                completion = openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo",
                                    messages=[{"role": "user", "content": prompt}],
                                    temperature=0,
                                )
                                llm_answer = completion["choices"][0]["message"]["content"].lower()
                            elif cfg.openai_model == "text-davinci-003":
                                completion = openai.Completion.create(
                                    model="text-davinci-003",
                                    prompt=prompt,
                                    temperature=0,
                                )
                                llm_answer = completion["choices"][0].text.strip().lower()
                            else:
                                raise ValueError(f"Invalid model name: {cfg.openai_model}")
                            break
                        except openai.error.RateLimitError as e:
                            time.sleep(1)
                        except openai.error.APIError as e:
                            time.sleep(1)
                        except Exception as e:
                            raise e

                    # postprocess
                    options_pattern = "("
                    for c in sample["classes"][i]:
                        name = qa_classes[c.item()]
                        if "(" in name:
                            name = name[:name.find("(")] + "\\" + name[name.find("("):]
                        if ")" in name:
                            name = name[:name.find(")")] + "\\" + name[name.find(")"):]
                        options_pattern += name + ")|("
                    options_pattern += "none)"
                    llm_answer = set([x.group() for x in re.finditer(options_pattern, llm_answer)])

                    with open(os.path.join(subset, str(num) + ".txt"), "w") as f:
                        print(f"ECG IDs: {sample['ecg_id'][i][0]}, {sample['ecg_id'][i][1]}\n", file=f)
                        print(prompt, file=f)
                        print("Answer: ", end="", file=f)
                        print(llm_answer, file=f)
                        print("", file=f)
                        print("GT: ", end="", file=f)
                        print(answer, file=f)
                        print("", file=f)
                        if answer == llm_answer:
                            print("Score: 1", file=f)
                        else:
                            print("Score: 0", file=f)

                    num += 1
                    if answer == llm_answer:
                        correct["question_type1"][sample["question_type1"][i].item()] += 1
                        correct["question_type2"][sample["question_type2"][i].item()] += 1
                        correct["question_type3"][sample["question_type3"][i].item()] += 1
                        inner_correct["question_type1"][sample["question_type1"][i].item()] += 1
                        inner_correct["question_type2"][sample["question_type2"][i].item()] += 1
                        inner_correct["question_type3"][sample["question_type3"][i].item()] += 1
                    
                    total["question_type1"][sample["question_type1"][i].item()] += 1
                    total["question_type2"][sample["question_type2"][i].item()] += 1
                    total["question_type3"][sample["question_type3"][i].item()] += 1
                    inner_total["question_type1"][sample["question_type1"][i].item()] += 1
                    inner_total["question_type2"][sample["question_type2"][i].item()] += 1
                    inner_total["question_type3"][sample["question_type3"][i].item()] += 1

                    if num % cfg.common.log_interval == 0:
                        inner_acc = dict()
                        for key1 in inner_total.keys():
                            for key2 in inner_total[key1].keys():
                                inner_acc[f"{key1}_{key2}"] = (inner_correct[key1][key2] / inner_total[key1][key2]) if inner_total[key1][key2] > 0 else 0

                        if (
                            distributed_utils.is_master(cfg.distributed_training)
                            and cfg.common.wandb_project is not None
                            and cfg.common.wandb_entity is not None
                        ):
                            prefix = subset + "_inner/"
                            wandb_logs = {}
                            for key in inner_acc.keys():
                                wandb_logs[prefix + key + "_em_accuracy"] = inner_acc[key]
                            wandb.log(wandb_logs, step=num)
                        
                        inner_total = {"question_type1": dict(), "question_type2": dict(), "question_type3": dict()}
                        inner_correct = {"question_type1": dict(), "question_type2": dict(), "question_type3": dict()}

        acc = dict()
        for key1 in total.keys():
            for key2 in total[key1].keys():
                acc[f"{key1}_{key2}"] = (correct[key1][key2] / total[key1][key2]) if total[key1][key2] > 0 else 0
        if (
            distributed_utils.is_master(cfg.distributed_training)
            and cfg.common.wandb_project is not None
            and cfg.common.wandb_entity is not None
        ):
            prefix = subset + "/"
            wandb_logs = {}
            for key in acc.keys():
                wandb_logs[prefix + key + "_em_accuracy"] = acc[key]
            wandb.log(wandb_logs)
        else:
            for key, val in acc.items():
                print(f"{key}: {val:.4f}")

@hydra.main(config_path=os.path.join("config"), config_name = "config")
def hydra_main(cfg: Config, **kwargs) -> None:
    add_defaults(cfg)

    if cfg.common.reset_logging:
        reset_logging() # Hydra hijacks logging, fix that
    else:
        if HydraConfig.initialized():
            with open_dict(cfg):
                # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
                cfg.job_logging_cfg = OmegaConf.to_container(
                    HydraConfig.get().job_logging, resolve=True
                )

    with omegaconf_no_object_check():
        cfg = OmegaConf.create(
            OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        )
    OmegaConf.set_struct(cfg, True)

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    main(cfg, **kwargs)

def cli_main():
    try:
        from hydra._internal.utils import get_args

        cfg_name = get_args().config_name or "config"
    except:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "config"
    hydra_init(cfg_name)
    hydra_main()

if __name__ == "__main__":
    cli_main()