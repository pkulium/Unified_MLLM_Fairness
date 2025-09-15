import torch
import torch.distributed as dist

from torch import nn
from torch.utils.data import ConcatDataset, Dataset, DistributedSampler, Sampler
from transformers import Trainer
from transformers.trainer import ALL_LAYERNORM_LAYERS
from transformers.trainer import get_parameter_names, has_length, is_sagemaker_mp_enabled
from typing import List, Optional, Dict, Union, Tuple, Any

from vila_u.mm_utils import KeywordsStoppingCriteria
from vila_u.constants import IGNORE_INDEX
import torch.amp as amp
from typing import Any, Callable, Literal, Optional, Union
from transformers import  PreTrainedModel
import torch.nn.functional as F

def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]

    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    assert all(l != 0 for l in lengths), "Should not have zero length."

    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)

    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [
        lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)
    ]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class VILADistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        batch_size=None,
        sample_len_list=None,
        force_accumulation=True,
        chunk_sampler=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval" " [0, {}]".format(rank, num_replicas - 1)
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = True

        self.org_sample_len_list = self.per_replica_samples = sample_len_list
        assert sum(sample_len_list) == len(self.dataset)

        self.batch_size = batch_size
        self.global_batch_size = batch_size * num_replicas

        if self.drop_last:
            self.per_replica_samples = [
                sample_len // (self.num_replicas * batch_size) * batch_size for sample_len in self.per_replica_samples
            ]
            self.num_samples = sum(self.per_replica_samples)
        else:
            raise NotImplementedError

        self.total_size = self.num_samples * self.num_replicas
        
        self.total_samples = [samples * self.num_replicas for samples in self.per_replica_samples]

        self.shuffle = shuffle
        self.seed = seed

        self.force_accumulation = force_accumulation
        self.chunk_sampler = chunk_sampler

    def __iter__(self):
        import random

        indices = list(range(len(self.dataset)))

        indices_list = []
        for i in range(len(self.org_sample_len_list)):
            indices_list.append(
                indices[sum(self.org_sample_len_list[:i]) : sum(self.org_sample_len_list[:i]) + self.total_samples[i]]
            )

        assert sum([len(indices) for indices in indices_list]) == self.total_size, (
            sum([len(indices) for indices in indices_list]),
            self.total_size,
        )

        for idx, indices in enumerate(indices_list):
            indices_list[idx] = indices[
                self.rank * self.per_replica_samples[idx] : (self.rank + 1) * self.per_replica_samples[idx]
            ]

        random.seed(self.seed + self.epoch)
        for indice in range(len(indices_list)):
            if self.chunk_sampler:
                list_split = [indices_list[indice][i:i+1000] for i in range(0, len(indices_list[indice]), 1000)]
                for i in range(len(list_split)):
                    random.shuffle(list_split[i])
                random.shuffle(list_split)
                list_merge = []
                for li in list_split:
                    list_merge += li
                indices_list[indice] = list_merge
            else:
                random.shuffle(indices_list[indice])

        indices_list = sorted(indices_list, key=lambda x: -len(x))
        all_indices = [-1] * self.num_samples
        indices_available = list(range(self.num_samples))

        for indice in indices_list:
            original_indices = range(len(indice))
            transformed_indices = [idx * len(indices_available) // len(indice) for idx in original_indices]
            mapped_indices = [indices_available[idx] for idx in transformed_indices]
            for idx in reversed(transformed_indices):
                del indices_available[idx]
            for i, idx in enumerate(mapped_indices):
                all_indices[idx] = indice[i]
        assert -1 not in all_indices

        return iter(all_indices)


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)


class VILAUTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        sample_len_list = self.args.sample_lens
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        return VILADistributedSampler(
            self.train_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            seed=seed,
            batch_size=self.args.train_batch_size,
            sample_len_list=sample_len_list,
            chunk_sampler=self.args.chunk_sampler,
        )

        if self.args.group_by_modality_length:
            if not isinstance(self.train_dataset, ConcatDataset):
                lengths = self.train_dataset.modality_lengths
            else:
                lengths = []
                for d in self.train_dataset.datasets:
                    lengths += d.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        if self.eval_dataset is None or not has_length(self.eval_dataset):
            return None

        sample_len_list = self.args.eval_sample_lens
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        return VILADistributedSampler(
            eval_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            seed=seed,
            batch_size=self.args.eval_batch_size,
            sample_len_list=sample_len_list,
        )

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        if self.args.use_peft:
            # Freeze the base model parameters
            # for param in self.model.base_model.parameters():
                # param.requires_grad = False

            # Only LoRA parameters are trainable
            optimizer_grouped_parameters = [
                {
                    "params": [p for p in self.model.parameters() if p.requires_grad],
                    "lr": self.args.learning_rate,
                    "weight_decay": self.args.weight_decay,
                }
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            return self.optimizer
            
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def save_model(self, output_dir: Optional[str], _internal_call: bool):
        ## save tuned model separately
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            # TODO(ligeng): fix save_model for multi-node training on large models (e.g., Llama-70b)
            state_dict = self.model.state_dict()

        if self.args.should_save:
            return self.model.save_pretrained(output_dir, state_dict=state_dict)
    
    @torch.no_grad()
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.current_dataset_name == "tgif":
            assert inputs["input_ids"].shape[0] == 1
            stopping_criteria = KeywordsStoppingCriteria(["</s>"], self.model.tokenizer, inputs["input_ids"])
            B, L = inputs["input_ids"].shape
            generation_input_ids = []
            for i in range(B):
                input_id = inputs["input_ids"][i].clone()
                label_begin_idx = torch.nonzero(inputs["labels"][i]!=IGNORE_INDEX)[0, 0]
                generation_input_ids.append(input_id[:label_begin_idx])
            generation_input_ids = torch.stack(generation_input_ids, dim=0)
            generation_output_ids = self.model.generate(
                generation_input_ids,
                images=inputs["images"],
                do_sample=True,
                temperature=0.2,
                max_new_tokens=self.args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )
            outputs = self.model.tokenizer.batch_decode(generation_output_ids, skip_special_tokens=True)
            print(outputs)
            self.total_cnt += len(outputs)
            for output, answer in zip(outputs, inputs["generation_labels"]):
                self.match_cnt += output==answer

            return torch.tensor(0., device=inputs["input_ids"].device), None, None
        else:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

IGNORE_INDEX = -100
def get_batch_logps(
    logits: "torch.Tensor", labels: "torch.Tensor", label_pad_token_id: int = IGNORE_INDEX
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    r"""
    Computes the log probabilities of the given labels under the given logits.

    Returns:
        logps: A tensor of shape (batch_size,) containing the sum of log probabilities.
        valid_length: A tensor of shape (batch_size,) containing the number of non-masked tokens.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")

    labels = labels.clone()
    loss_mask = labels != label_pad_token_id
    labels[labels == label_pad_token_id] = 0  # dummy token
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

class VILAUBPOTrainer(VILAUTrainer):

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss for batched log probabilities of the policy model.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.args.beta * odds_ratio_loss
        return orpo_loss
    
    def bpo_loss(
        self,
        pref1_logps: torch.Tensor,
        pref2_logps: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Computes a modified loss that:
        1) Applies supervised fine-tuning (SFT) loss to both "pref1" and "pref2".
        2) Encourages balanced odds between pref1 and pref2 using a sigmoid-based penalty.

        Args:
            pref1_logps: log probabilities [batch] for "pref1".
            pref2_logps: log probabilities [batch] for "pref2".

        Returns:
            A scalar loss (after .mean()).
        """
        # ------------------------------------------------
        # 1) SFT Loss for Both Preferences
        # ------------------------------------------------
        # Example: we penalize negative log-probabilities for each category.
        # Depending on your data/labels, you might do something more nuanced.
        # sft_loss_pref1 = -pref1_logps          # -log p(pref1)
        # sft_loss_pref2 = -pref2_logps          # -log p(pref2)
        # sft_loss = sft_loss_pref1 + sft_loss_pref2

        # ------------------------------------------------
        # 2) Compute Log-Odds for Each Preference
        # ------------------------------------------------
        # log_odds(pref) = log p(pref) - log(1 - p(pref))
        # We'll then take the difference for pref1 vs. pref2.
        pref1_log_odds = pref1_logps - torch.log1p(-torch.exp(pref1_logps))
        pref2_log_odds = pref2_logps - torch.log1p(-torch.exp(pref2_logps))
        log_odds_diff = pref1_log_odds - pref2_log_odds

        # ------------------------------------------------
        # 3) Balanced Odds Penalty (Sigmoid-Based)
        # ------------------------------------------------
        # We want log_odds_diff ~ 0 => p ~ 0.5.
        # So we take sigmoid(log_odds_diff), which maps diff -> (0,1).
        # Then measure how far it is from 0.5. Perfect balance => ~0 penalty.
        balanced_prob = torch.sigmoid(log_odds_diff)         # in (0,1)
        balanced_odds_loss = (balanced_prob - 0.5).pow(2)    # penalize deviation from 0.5

        # ------------------------------------------------
        # 4) Combine All Loss Terms
        # ------------------------------------------------
        total_loss = self.args.beta * balanced_odds_loss

        # return total_loss.mean()
        return total_loss
    
    def bpo_loss_v1(
        self,
        pref1_logps: torch.Tensor,
        pref2_logps: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Computes a modified loss that:
        1) Applies supervised fine-tuning (SFT) loss to both "pref1" and "pref2".
        2) Encourages balanced odds between pref1 and pref2 using a sigmoid-based penalty.

        Args:
            pref1_logps: log probabilities [batch] for "pref1".
            pref2_logps: log probabilities [batch] for "pref2".

        Returns:
            A scalar loss (after .mean()).
        """
        # ------------------------------------------------
        # 1) SFT Loss for Both Preferences
        # ------------------------------------------------
        # Example: we penalize negative log-probabilities for each category.
        # Depending on your data/labels, you might do something more nuanced.
        # sft_loss_pref1 = -pref1_logps          # -log p(pref1)
        # sft_loss_pref2 = -pref2_logps          # -log p(pref2)
        # sft_loss = sft_loss_pref1 + sft_loss_pref2

        # ------------------------------------------------
        # 2) Compute Log-Odds for Each Preference
        # ------------------------------------------------
        # log_odds(pref) = log p(pref) - log(1 - p(pref))
        # We'll then take the difference for pref1 vs. pref2.
        pref1_log_odds = pref1_logps - torch.log1p(-torch.exp(pref1_logps))
        pref2_log_odds = pref2_logps - torch.log1p(-torch.exp(pref2_logps))
        log_odds_diff = pref1_log_odds - pref2_log_odds

        # ------------------------------------------------
        # 3) Balanced Odds Penalty (Sigmoid-Based)
        # ------------------------------------------------q
        # We want log_odds_diff ~ 0 => p ~ 0.5.
        # So we take sigmoid(log_odds_diff), which maps diff -> (0,1).
        # Then measure how far it is from 0.5. Perfect balance => ~0 penalty.
        balanced_prob = torch.sigmoid(log_odds_diff)         # in (0,1)
        balanced_odds_loss = torch.log1p((balanced_prob - 0.5).pow(2))

        # ------------------------------------------------
        # 4) Combine All Loss Terms
        # ------------------------------------------------
        total_loss = self.args.beta * balanced_odds_loss

        return total_loss.mean()

    def bpo_loss_v2(
        self,
        pref1_logps: torch.Tensor,
        pref2_logps: torch.Tensor,
        threshold: float = 0.3,
        extra_weight: float = 1.0
    ) -> torch.Tensor:
        r"""
        Computes a modified loss that:
        (1) Applies a balanced-odds penalty (sigmoid-based).
        (2) If log_odds_diff is strongly positive (> threshold), 
            we add pref1_log_odds. If it's strongly negative (< -threshold),
            we add pref2_log_odds. Otherwise, 0.

        Args:
            pref1_logps: log probabilities [batch] for "pref1".
            pref2_logps: log probabilities [batch] for "pref2".
            threshold: cutoff value for "very positive" or "very negative".
            extra_weight: scaling factor for the extra penalty.

        Returns:
            A scalar loss (after .mean()).
        """
        # ------------------------------------------------
        # 1) Balanced Odds Penalty (Original)
        # ------------------------------------------------
        # log_odds(pref) = log p(pref) - log(1 - p(pref))
        log_odds_diff = torch.sigmoid(pref1_logps - pref2_logps) - 0.5

        # ------------------------------------------------
        # 2) Extra Penalty for "Very Positive" or "Very Negative"
        # ------------------------------------------------
        # If log_odds_diff[i] > +threshold => add pref2_log_odds[i]
        # If log_odds_diff[i] < -threshold => add pref1_log_odds[i]
        # else => 0
        pos_mask = (log_odds_diff > threshold)
        neg_mask = (log_odds_diff < -threshold)
        # 0 for the "ignore" region where |diff| <= threshold

        # Build the extra penalty array
        extra_penalty = torch.zeros_like(log_odds_diff)  # [batch]

        pref1_extra_penalty = F.logsigmoid(pref1_logps)
        pref2_extra_penalty = F.logsigmoid(pref2_logps)
        extra_penalty[pos_mask] = pref2_extra_penalty[pos_mask]
        extra_penalty[neg_mask] = pref1_extra_penalty[neg_mask]
        # e.g., extra_penalty might be positive or negative depending
        # on the sign of prefX_log_odds. If you want strictly positive
        # penalty, you could do e.g. +torch.abs(...) or something else.

        # Scale the penalty if desired
        extra_penalty = - extra_weight * extra_penalty

        # Average over the batch dimension
        return extra_penalty.mean()
    
    def bpo_loss_v3(
        self,
        pref1_logps: torch.Tensor,
        pref2_logps: torch.Tensor,
        threshold: float = 0.3,
        extra_weight: float = 1.0
    ) -> torch.Tensor:
        r"""
        Computes a modified loss that:
        (1) Applies a balanced-odds penalty (sigmoid-based).
        (2) If log_odds_diff is strongly positive (> threshold), 
            we add pref1_log_odds. If it's strongly negative (< -threshold),
            we add pref2_log_odds. Otherwise, 0.

        Args:
            pref1_logps: log probabilities [batch] for "pref1".
            pref2_logps: log probabilities [batch] for "pref2".
            threshold: cutoff value for "very positive" or "very negative".
            extra_weight: scaling factor for the extra penalty.

        Returns:
            A scalar loss (after .mean()).
        """
        # ------------------------------------------------
        # 1) Balanced Odds Penalty (Original)
        # ------------------------------------------------
        # log_odds(pref) = log p(pref) - log(1 - p(pref))
        # We'll compute the difference for pref1 vs. pref2
        pref1_log_odds = pref1_logps - torch.log1p(-torch.exp(pref1_logps))
        pref2_log_odds = pref2_logps - torch.log1p(-torch.exp(pref2_logps))
        log_odds_diff = pref1_log_odds - pref2_log_odds
        log_odds_diff = torch.sigmoid(log_odds_diff) - 0.5         # in (-0.5,0.5)

        # ------------------------------------------------
        # 2) Extra Penalty for "Very Positive" or "Very Negative"
        # ------------------------------------------------
        # If log_odds_diff[i] > +threshold => add pref2_log_odds[i]
        # If log_odds_diff[i] < -threshold => add pref1_log_odds[i]
        # else => 0
        pos_mask = (log_odds_diff > threshold)
        neg_mask = (log_odds_diff < -threshold)
        # 0 for the "ignore" region where |diff| <= threshold

        # Build the extra penalty array
        extra_penalty = torch.zeros_like(log_odds_diff)  # [batch]

        extra_penalty[pos_mask] = pref2_logps[pos_mask]
        extra_penalty[neg_mask] = pref1_logps[neg_mask]
        # e.g., extra_penalty might be positive or negative depending
        # on the sign of prefX_log_odds. If you want strictly positive
        # penalty, you could do e.g. +torch.abs(...) or something else.

        # Scale the penalty if desired
        extra_penalty = - extra_weight * extra_penalty

        # Average over the batch dimension
        return extra_penalty.mean()
    
   
   

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes SimPO loss for batched log probabilities of the policy model.
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.args.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.args.beta * logits)
        return simpo_loss

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for preference learning.
        """
        if not self.args.use_ref_model:
            if self.args.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.args.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.args.loss_type == "bpo":
                losses = self.bpo_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.args.loss_type == "bpo_v1":
                losses = self.bpo_loss_v1(policy_chosen_logps, policy_rejected_logps) 
            elif self.args.loss_type == "bpo_v2":
                losses = self.bpo_loss_v2(policy_chosen_logps, policy_rejected_logps, extra_weight = self.args.beta)
            elif self.args.loss_type == "bpo_v3":
                losses = self.bpo_loss_v3(policy_chosen_logps, policy_rejected_logps, extra_weight = self.args.beta)
            else:
                raise NotImplementedError(f"Unknown loss type: {self.args.loss_type}.")

            chosen_rewards = self.args.beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = self.args.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )

        return losses, chosen_rewards, rejected_rewards

    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """
        if self.args.use_ref_model:
            batch = {k: v.detach().clone() for k, v in batch.items()}  # avoid error
        
        # Convert the lengths to a tensor on the same device as input_ids
        repeat_lengths = torch.tensor([len(img_list) for img_list in batch['images']],
                                    device=batch['input_ids'].device)
                                    
        # Now repeat with tensors on the same device
        input_ids = batch['input_ids'].repeat_interleave(repeat_lengths, dim=0)
        labels = batch['labels'].repeat_interleave(repeat_lengths, dim=0)
        attention_mask = batch['attention_mask'].repeat_interleave(repeat_lengths, dim=0)

        # Handle images - make sure they're on the same device
        images = torch.cat([img for img_list in batch['images'] for img in img_list])
        images = images.to(batch['input_ids'].device)

        # Update batch in-place
        batch['input_ids'] = input_ids
        batch['labels'] = labels 
        batch['attention_mask'] = attention_mask
        batch['images'] = images

        output = model(**batch, return_dict=True, use_cache=False)
        all_logits = output.image_logits.to(torch.float32)
        all_labels = output.image_labels
        self.all_logits = all_logits
        self.all_labels = all_labels
        # import pdb 
        # pdb.set_trace()
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=all_labels)
        if 'bpo' in self.args.loss_type or self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length, output

    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Computes log probabilities of the reference model.
        """
        if not self.args.use_ref_model:
            return None, None

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(ref_model, batch)

        return reference_chosen_logps, reference_rejected_logps

    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
            output
        ) = self.concatenated_forward(model, batch)

        if self.args.beta == 0:
            return output.loss, metrics

        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        losses += output.loss
        return losses.mean(), metrics


    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        current_epoch = self.state.epoch
    
        if current_epoch < self.args.bpo_start_epoch:
            self.args.beta = 0

        compute_loss_context_manager = amp.autocast("cuda") 
        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)

        if return_outputs:
            return loss, metrics

        return loss
    
    @torch.no_grad()
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.current_dataset_name == "tgif":
            assert inputs["input_ids"].shape[0] == 1
            stopping_criteria = KeywordsStoppingCriteria(["</s>"], self.model.tokenizer, inputs["input_ids"])
            B, L = inputs["input_ids"].shape
            generation_input_ids = []
            for i in range(B):
                input_id = inputs["input_ids"][i].clone()
                label_begin_idx = torch.nonzero(inputs["labels"][i]!=IGNORE_INDEX)[0, 0]
                generation_input_ids.append(input_id[:label_begin_idx])
            generation_input_ids = torch.stack(generation_input_ids, dim=0)
            generation_output_ids = self.model.generate(
                generation_input_ids,
                images=inputs["images"],
                do_sample=True,
                temperature=0.2,
                max_new_tokens=self.args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )
            outputs = self.model.tokenizer.batch_decode(generation_output_ids, skip_special_tokens=True)
            print(outputs)
            self.total_cnt += len(outputs)
            for output, answer in zip(outputs, inputs["generation_labels"]):
                self.match_cnt += output==answer

            return torch.tensor(0., device=inputs["input_ids"].device), None, None
        else:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)



# class VILAUSharpnessTrainer(VILAUTrainer):
#     def __init__(self, *args, lambda_norm: float = 1.0, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.lambda_norm = lambda_norm  # Hyperparameter for the norm difference loss

#     def compute_loss(
#         self,
#         model: Union[PreTrainedModel, nn.Module],
#         inputs: dict[str, Union[torch.Tensor, Any]],
#         return_outputs=False,
#         num_items_in_batch=None,
#     ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
#         """
#         Compute a combined loss that includes the original loss (e.g. cross-entropy) 
#         and an additional term to minimize the difference in gradient norm sums between 
#         even- and odd-indexed samples.

#         The additional loss term is defined as:
#             L_norm = | sum_{even i} ||∇_θ L_i||₂² - sum_{odd i} ||∇_θ L_i||₂² |

#         and the total loss is:
#             L_total = L_original + λ * L_norm

#         Parameters:
#             model: The transformer model.
#             inputs: Dictionary containing model inputs (e.g. input_ids, attention_mask, labels, etc.).
#             return_outputs: Whether to return model outputs along with loss.
#             num_items_in_batch: (Optional) number of items in batch. If None, it is derived from the input tensor.

#         Returns:
#             Either the combined loss (tensor) or a tuple of (loss, outputs).
#         """

#         # Forward pass to obtain the original loss.
#         # It is assumed that the model returns an output object with a 'loss' attribute.
#         outputs = model(**inputs)
#         loss_original = outputs.loss

#         # --- Compute per-sample gradient norms on normalization parameters ---
#         # We filter parameters that belong to normalization layers.
#         norm_params = [
#             param for name, param in model.named_parameters() 
#             if "norm" in name and param.requires_grad
#         ]
#         if not norm_params:
#             raise ValueError("No normalization layer parameters found in model.")

#         # We assume the batch size is the size of the first dimension of one of the input tensors.
#         any_tensor = next((v for v in inputs.values() if isinstance(v, Tensor)), None)
#         if any_tensor is None:
#             raise ValueError("No input tensor found in inputs.")
#         batch_size = num_items_in_batch if num_items_in_batch is not None else any_tensor.size(0)

#         # List to store squared L2 norms of gradients for each sample.
#         grad_norms = []

#         # We compute per-sample loss and then, via autograd, obtain gradients on norm_params.
#         # Set create_graph=True to allow differentiating through these gradients.
#         for i in range(batch_size):
#             # Prepare input for a single sample.
#             # For tensor inputs, select the i-th element and add a batch dimension.
#             sample_inputs = {
#                 key: (value[i].unsqueeze(0) if isinstance(value, Tensor) else value)
#                 for key, value in inputs.items()
#             }
#             # Forward pass for the single sample.
#             sample_outputs = model(**sample_inputs)
#             sample_loss = sample_outputs.loss

#             # Compute gradients for the normalization parameters for this sample.
#             # Using create_graph=True so that gradients of L_norm can propagate.
#             grads = torch.autograd.grad(
#                 sample_loss, norm_params, retain_graph=True, create_graph=True, allow_unused=True
#             )
#             # Compute squared L2 norm for this sample.
#             sample_grad_norm_sq = 0.0
#             for grad in grads:
#                 if grad is not None:
#                     sample_grad_norm_sq += torch.sum(grad ** 2)
#             grad_norms.append(sample_grad_norm_sq)

#         # Separate even and odd indexed samples and compute sums of their squared gradient norms.
#         even_norm_sum = sum(grad_norms[i] for i in range(0, batch_size, 2))
#         odd_norm_sum = sum(grad_norms[i] for i in range(1, batch_size, 2))
#         # The additional loss term is the absolute difference between these two sums.
#         L_norm = torch.abs(even_norm_sum - odd_norm_sum)

#         # --- Combined loss ---
#         total_loss = loss_original + self.lambda_norm * L_norm

#         if return_outputs:
#             return total_loss, outputs

#         return total_loss


from typing import Union, Any, Optional
import torch
from torch import nn, Tensor
from transformers import PreTrainedModel

class VILAUSharpnessTrainer(VILAUTrainer):
    def __init__(self, *args, lambda_norm: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_norm = lambda_norm

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        """
        Compute combined loss including original loss and gradient norm difference between even/odd samples.
        Performs only a single forward pass and computes per-sample gradients from the resulting loss.

        Args:
            model: The transformer model
            inputs: Dictionary of model inputs
            return_outputs: Whether to return model outputs along with loss
            num_items_in_batch: Optional explicit batch size

        Returns:
            Either the combined loss tensor or a tuple of (loss, outputs)

        Raises:
            ValueError: If no normalization parameters are found or input tensors are invalid
        """
        # Convert the lengths to a tensor on the same device as input_ids
        repeat_lengths = torch.tensor([len(img_list) for img_list in inputs['images']],
                                    device=inputs['input_ids'].device)

        # Now repeat with tensors on the same device
        input_ids = inputs['input_ids'].repeat_interleave(repeat_lengths, dim=0)
        labels = inputs['labels'].repeat_interleave(repeat_lengths, dim=0)
        attention_mask = inputs['attention_mask'].repeat_interleave(repeat_lengths, dim=0)

        # Handle images - make sure they're on the same device
        images = torch.cat([img for img_list in inputs['images'] for img in img_list])
        images = images.to(inputs['input_ids'].device)

        # Update inputs in-place
        inputs['input_ids'] = input_ids
        inputs['labels'] = labels 
        inputs['attention_mask'] = attention_mask
        inputs['images'] = images

        outputs = model(**inputs, return_dict=True, use_cache=False)

        loss_original = outputs.loss

        # Get normalization parameters
        norm_params = [
            param for name, param in model.named_parameters()
            if param.requires_grad
        ]
        if not norm_params:
            raise ValueError("No normalization layer parameters found in model")

        # Get batch size
        batch_size = (num_items_in_batch if num_items_in_batch is not None 
                     else next(v.size(0) for v in inputs.values() if isinstance(v, Tensor)))

        # Get per-sample losses
        per_sample_losses = self._compute_per_sample_losses(outputs, batch_size)

        # Initialize gradient norm sums on the correct device
        device = per_sample_losses.device
        even_norm_sum = torch.tensor(0.0, device=device)
        odd_norm_sum = torch.tensor(0.0, device=device)

        # Compute gradient norms for even and odd samples
        for i in range(batch_size):
            # Compute gradients for this sample's loss
            grads = torch.autograd.grad(
                per_sample_losses[i],
                norm_params,
                retain_graph=True,
                create_graph=True,
                allow_unused=True
            )
            
            # Compute squared L2 norm for this sample
            sample_grad_norm = sum(
                torch.sum(grad ** 2) if grad is not None else 0.0
                for grad in grads
            )

            # Add to appropriate sum
            if i % 2 == 0:
                even_norm_sum += sample_grad_norm
            else:
                odd_norm_sum += sample_grad_norm

        # Calculate norm difference loss and combine with original loss
        norm_loss = torch.abs(even_norm_sum - odd_norm_sum)
        total_loss = loss_original + self.lambda_norm * norm_loss

        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_per_sample_losses(
        self, 
        outputs: Any, 
        batch_size: int
    ) -> torch.Tensor:
        """
        Extract per-sample losses from the model outputs.
        This method should be customized based on your specific model's output format.
        """
        # If the model returns per-sample losses directly
        if hasattr(outputs, "per_sample_losses"):
            return outputs.per_sample_losses
            
        # If using a language model with logits output
        elif hasattr(outputs, "image_logits") and hasattr(outputs, "image_labels"):
            # Assuming a standard language modeling setup
            # This needs to be adjusted based on your specific loss function
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            logits = outputs.image_logits.view(-1, outputs.image_logits.size(-1))
            labels = outputs.image_labels.view(-1)
            per_token_losses = loss_fct(logits, labels)
            
            # Reshape and reduce to get per-sample losses
            return per_token_losses.view(batch_size, -1).mean(dim=1)
            
        else:
            raise ValueError(
                "Could not compute per-sample losses. "
                "Model outputs must either provide per_sample_losses "
                "or contain logits and labels."
            )

class VILAUBPOLOSS:
    def __init__(self, loss_type = "bpo_v1", beta = 0.1, threshold = 0.3):
        self.loss_type = loss_type
        self.beta = beta
        self.threshold = threshold

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss for batched log probabilities of the policy model.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss
    
    def bpo_loss_v1(
        self,
        pref1_logps: torch.Tensor,
        pref2_logps: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Computes a modified loss that:
        1) Applies supervised fine-tuning (SFT) loss to both "pref1" and "pref2".
        2) Encourages balanced odds between pref1 and pref2 using a sigmoid-based penalty.

        Args:
            pref1_logps: log probabilities [batch] for "pref1".
            pref2_logps: log probabilities [batch] for "pref2".

        Returns:
            A scalar loss (after .mean()).
        """
        # ------------------------------------------------
        # 1) SFT Loss for Both Preferences
        # ------------------------------------------------
        # Example: we penalize negative log-probabilities for each category.
        # Depending on your data/labels, you might do something more nuanced.
        # sft_loss_pref1 = -pref1_logps          # -log p(pref1)
        # sft_loss_pref2 = -pref2_logps          # -log p(pref2)
        # sft_loss = sft_loss_pref1 + sft_loss_pref2

        # ------------------------------------------------
        # 2) Compute Log-Odds for Each Preference
        # ------------------------------------------------
        # log_odds(pref) = log p(pref) - log(1 - p(pref))
        # We'll then take the difference for pref1 vs. pref2.
        pref1_log_odds = pref1_logps - torch.log1p(-torch.exp(pref1_logps))
        pref2_log_odds = pref2_logps - torch.log1p(-torch.exp(pref2_logps))
        log_odds_diff = pref1_log_odds - pref2_log_odds

        # ------------------------------------------------
        # 3) Balanced Odds Penalty (Sigmoid-Based)
        # ------------------------------------------------q
        # We want log_odds_diff ~ 0 => p ~ 0.5.
        # So we take sigmoid(log_odds_diff), which maps diff -> (0,1).
        # Then measure how far it is from 0.5. Perfect balance => ~0 penalty.
        balanced_prob = torch.sigmoid(log_odds_diff)         # in (0,1)
        balanced_odds_loss = torch.log1p((balanced_prob - 0.5).pow(2))

        # ------------------------------------------------
        # 4) Combine All Loss Terms
        # ------------------------------------------------
        total_loss = self.beta * balanced_odds_loss

        return total_loss.mean()

    def bpo_loss_v2(
        self,
        pref1_logps: torch.Tensor,
        pref2_logps: torch.Tensor,
        threshold: float = 0.3,
        extra_weight: float = 1.0
    ) -> torch.Tensor:
        r"""
        Computes a modified loss that:
        (1) Applies a balanced-odds penalty (sigmoid-based).
        (2) If log_odds_diff is strongly positive (> threshold), 
            we add pref1_log_odds. If it's strongly negative (< -threshold),
            we add pref2_log_odds. Otherwise, 0.

        Args:
            pref1_logps: log probabilities [batch] for "pref1".
            pref2_logps: log probabilities [batch] for "pref2".
            threshold: cutoff value for "very positive" or "very negative".
            extra_weight: scaling factor for the extra penalty.

        Returns:
            A scalar loss (after .mean()).
        """
        # ------------------------------------------------
        # 1) Balanced Odds Penalty (Original)
        # ------------------------------------------------
        # log_odds(pref) = log p(pref) - log(1 - p(pref))
        log_odds_diff = torch.sigmoid(pref1_logps - pref2_logps) - 0.5

        # ------------------------------------------------
        # 2) Extra Penalty for "Very Positive" or "Very Negative"
        # ------------------------------------------------
        # If log_odds_diff[i] > +threshold => add pref2_log_odds[i]
        # If log_odds_diff[i] < -threshold => add pref1_log_odds[i]
        # else => 0
        pos_mask = (log_odds_diff > threshold)
        neg_mask = (log_odds_diff < -threshold)
        # 0 for the "ignore" region where |diff| <= threshold

        # Build the extra penalty array
        extra_penalty = torch.zeros_like(log_odds_diff)  # [batch]

        pref1_extra_penalty = F.logsigmoid(pref1_logps)
        pref2_extra_penalty = F.logsigmoid(pref2_logps)
        extra_penalty[pos_mask] = pref2_extra_penalty[pos_mask]
        extra_penalty[neg_mask] = pref1_extra_penalty[neg_mask]
        # e.g., extra_penalty might be positive or negative depending
        # on the sign of prefX_log_odds. If you want strictly positive
        # penalty, you could do e.g. +torch.abs(...) or something else.

        # Scale the penalty if desired
        extra_penalty = - extra_weight * extra_penalty

        # Average over the batch dimension
        return extra_penalty.mean()
    
    def bpo_loss_v3(
        self,
        pref1_logps: torch.Tensor,
        pref2_logps: torch.Tensor,
        threshold: float = 0.3,
        extra_weight: float = 1.0
    ) -> torch.Tensor:
        r"""
        Computes a modified loss that:
        (1) Applies a balanced-odds penalty (sigmoid-based).
        (2) If log_odds_diff is strongly positive (> threshold), 
            we add pref1_log_odds. If it's strongly negative (< -threshold),
            we add pref2_log_odds. Otherwise, 0.

        Args:
            pref1_logps: log probabilities [batch] for "pref1".
            pref2_logps: log probabilities [batch] for "pref2".
            threshold: cutoff value for "very positive" or "very negative".
            extra_weight: scaling factor for the extra penalty.

        Returns:
            A scalar loss (after .mean()).
        """
        # ------------------------------------------------
        # 1) Balanced Odds Penalty (Original)
        # ------------------------------------------------
        # log_odds(pref) = log p(pref) - log(1 - p(pref))
        # We'll compute the difference for pref1 vs. pref2
        pref1_log_odds = pref1_logps - torch.log1p(-torch.exp(pref1_logps))
        pref2_log_odds = pref2_logps - torch.log1p(-torch.exp(pref2_logps))
        log_odds_diff = pref1_log_odds - pref2_log_odds
        log_odds_diff = torch.sigmoid(log_odds_diff) - 0.5         # in (-0.5,0.5)

        # ------------------------------------------------
        # 2) Extra Penalty for "Very Positive" or "Very Negative"
        # ------------------------------------------------
        # If log_odds_diff[i] > +threshold => add pref2_log_odds[i]
        # If log_odds_diff[i] < -threshold => add pref1_log_odds[i]
        # else => 0
        pos_mask = (log_odds_diff > threshold)
        neg_mask = (log_odds_diff < -threshold)
        # 0 for the "ignore" region where |diff| <= threshold

        # Build the extra penalty array
        extra_penalty = torch.zeros_like(log_odds_diff)  # [batch]

        pref1_extra_penalty = F.logsigmoid(pref1_logps)
        pref2_extra_penalty = F.logsigmoid(pref2_logps)
        extra_penalty[pos_mask] = pref2_extra_penalty[pos_mask]
        extra_penalty[neg_mask] = pref1_extra_penalty[neg_mask]
        # e.g., extra_penalty might be positive or negative depending
        # on the sign of prefX_log_odds. If you want strictly positive
        # penalty, you could do e.g. +torch.abs(...) or something else.

        # Scale the penalty if desired
        extra_penalty = - extra_weight * extra_penalty

        # Average over the batch dimension
        return extra_penalty.mean()
    
    def bpo_loss_v4(
        self,
        pref1_logps: torch.Tensor,
        pref2_logps: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Computes a modified loss that:
        1) Applies supervised fine-tuning (SFT) loss to both "pref1" and "pref2".
        2) Encourages balanced odds between pref1 and pref2 using a sigmoid-based penalty.

        Args:
            pref1_logps: log probabilities [batch] for "pref1".
            pref2_logps: log probabilities [batch] for "pref2".

        Returns:
            A scalar loss (after .mean()).
        """
        # ------------------------------------------------
        # 1) SFT Loss for Both Preferences
        # ------------------------------------------------
        # Example: we penalize negative log-probabilities for each category.
        # Depending on your data/labels, you might do something more nuanced.
        # sft_loss_pref1 = -pref1_logps          # -log p(pref1)
        # sft_loss_pref2 = -pref2_logps          # -log p(pref2)
        # sft_loss = sft_loss_pref1 + sft_loss_pref2

        # ------------------------------------------------
        # 2) Compute Log-Odds for Each Preference
        # ------------------------------------------------
        # log_odds(pref) = log p(pref) - log(1 - p(pref))
        # We'll then take the difference for pref1 vs. pref2.
        log_odds_diff = pref1_logps - pref2_logps


        # ------------------------------------------------
        # 3) Balanced Odds Penalty (Sigmoid-Based)
        # ------------------------------------------------q
        # Then measure how far it is from 0.5. Perfect balance => ~0 penalty.
        balanced_prob = torch.sigmoid(log_odds_diff)         # in (0,1)
        balanced_odds_loss = torch.log1p((balanced_prob - 0.5).pow(2))

        # ------------------------------------------------
        # 4) Combine All Loss Terms
        # ------------------------------------------------
        total_loss = self.beta * balanced_odds_loss

        return total_loss.mean()
    
    
    def bpo_loss_v5(
        self,
        pref1_logps: torch.Tensor,
        pref2_logps: torch.Tensor,
        threshold: float = 0.3,
        extra_weight: float = 1.0,
        logts_and_labels=None
    ) -> torch.Tensor:
        r"""
        Computes a modified loss that:
        (1) Applies a "balanced-odds" penalty (sigmoid-based).
        (2) If log_odds_diff is strongly positive (> threshold), 
            we add the pref2 penalty. If it is strongly negative (< -threshold),
            we add the pref1 penalty. Otherwise, 0.

        Args:
            pref1_logps: log probabilities [batch] for "pref1".
            pref2_logps: log probabilities [batch] for "pref2".
            threshold: cutoff value for "very positive" or "very negative".
            extra_weight: scaling factor for the extra penalty.
            logts_and_labels: tuple of (pref1_logits, pref2_logits, 
                                    pref1_labels, pref2_labels) used 
                                    for cross-entropy calculations

        Returns:
            A scalar loss (after .mean()) representing this extra penalty.
        """
        # prepare the logits and labels
        image_logits, image_labels = logts_and_labels
        B, seq_len, D, C = image_logits.shape
        image_logits = image_logits.reshape(B * seq_len * D, C)
        image_labels = image_labels.reshape(B * seq_len * D)
        image_labels = image_labels.clone()

        loss_fct = nn.CrossEntropyLoss(reduction='none')

        # Compute per-token loss (no reduction)
        per_token_loss = loss_fct(image_logits, image_labels)  # shape: [B*seq_len*D]

        # Reshape to [B, seq_len, D]
        per_token_loss = per_token_loss.view(B, seq_len, D)

        # Now you can average (or sum) across seq_len and D to get one loss per sample
        loss_per_sample = per_token_loss.mean(dim=[1, 2])  # shape: [B]
        pref1_extra_penalty, pref2_extra_penalty = loss_per_sample.split(B // 2, dim=0)

        # ------------------------------------------------
        # 1) Balanced Odds "Penalty" (using a sigmoided difference)
        # ------------------------------------------------
        # We interpret "log_odds_diff" as how much pref1 outclasses pref2.
        # NOTE: This is named 'log_odds_diff', but strictly it's the difference
        #       from 0.5 if we pass (pref1_logps - pref2_logps) through sigmoid.
        #       i.e. log_odds_diff > 0 means pref1 is larger than pref2.
        log_odds_diff = torch.sigmoid(pref1_logps - pref2_logps) - 0.5
        log_odds_diff_square = log_odds_diff.pow(2)

        # ------------------------------------------------
        # 2) Extra Penalty for "Very Positive" or "Very Negative"
        # ------------------------------------------------
        # If log_odds_diff[i] > +threshold => add pref2 penalty
        # If log_odds_diff[i] < -threshold => add pref1 penalty
        # Else => 0
        pos_mask = (log_odds_diff > threshold)
        neg_mask = (log_odds_diff < -threshold)

        # Prepare an extra penalty tensor (batch-sized)
        extra_penalty = torch.zeros_like(log_odds_diff)

        # ------------------------------------------------
        # Cross-Entropy for Each Preference
        # ------------------------------------------------
        # Apply them conditionally
        extra_penalty[pos_mask] = pref2_extra_penalty[pos_mask]
        extra_penalty[neg_mask] = pref1_extra_penalty[neg_mask]

        # Scale (and sign) the penalty if desired; for instance:
        #   - We multiply by -1 here, which *subtracts* the penalty from
        #     any larger objective you'd be maximizing (or equivalently
        #     *adds* it to a loss we are minimizing). Adjust as needed.
        # extra_penalty = extra_penalty * log_odds_diff_square
        extra_penalty = extra_weight * extra_penalty

        # Finally, reduce over batch
        return extra_penalty.mean()

    
    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes SimPO loss for batched log probabilities of the policy model.
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss

    def __call__(
        self,
        all_logits: "torch.Tensor",
        all_labels: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for preference learning.
        """
        all_logps, valid_length = self.get_batch_logps_image(all_logits, all_labels)
        if 'bpo' in self.loss_type or self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = all_logits.size(0) // 2
        policy_chosen_logps, policy_rejected_logps = all_logps.split(batch_size, dim=0)
        # chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        # chosen_labels, reject_labels = all_labels.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)

        if self.loss_type == "orpo":
            losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
        elif self.loss_type == "simpo":
            losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
        elif self.loss_type == "bpo_v1":
            losses = self.bpo_loss_v1(policy_chosen_logps, policy_rejected_logps) 
        elif self.loss_type == "bpo_v2":
            losses = self.bpo_loss_v2(policy_chosen_logps, policy_rejected_logps, threshold = self.threshold, extra_weight = self.beta)
        elif self.loss_type == "bpo_v3":
            losses = self.bpo_loss_v3(policy_chosen_logps, policy_rejected_logps, threshold = self.threshold, extra_weight = self.beta)
        elif self.loss_type == "bpo_v4":
            losses = self.bpo_loss_v4(policy_chosen_logps, policy_rejected_logps)
        elif self.loss_type == "bpo_v5":
            logts_and_labels = [all_logits, all_labels]
            losses = self.bpo_loss_v5(policy_chosen_logps, policy_rejected_logps, threshold = self.threshold, extra_weight = self.beta, logts_and_labels = logts_and_labels)
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")

        chosen_rewards = self.beta * policy_chosen_logps
        rejected_rewards = self.beta * policy_rejected_logps

        return losses, chosen_rewards, rejected_rewards
    

    def get_batch_logps_image(
        self, 
        image_logits: torch.Tensor,
        image_labels: torch.Tensor,
        label_pad_token_id: int = -100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes sum of log probabilities for each sample in the batch, ignoring pad tokens.
        
        Expects:
        image_logits shape = (B, seq_len, D, C)  [often B = 2 * real_batch_size]
        image_labels shape = (B, seq_len, D)
        Returns:
        logps_per_batch: (B,) sum of log probabilities of all valid tokens per sample
        valid_length:    (B,) count of valid (non-pad) tokens per sample
        """
        # 1) Unpack shapes
        B, seq_len, D, C = image_logits.shape
        
        # 2) Flatten
        flat_logits = image_logits.view(B * seq_len * D, C)       # => shape (B*seq_len*D, C)
        flat_labels = image_labels.view(B * seq_len * D)          # => shape (B*seq_len*D,)

        # 3) Create loss mask for pad tokens
        loss_mask = (flat_labels != label_pad_token_id)

        # 4) Replace pad tokens with some dummy index (0)
        flat_labels = flat_labels.clone()
        flat_labels[~loss_mask] = 0

        # 5) Compute log-softmax and gather the log-prob of the correct label
        log_probs = F.log_softmax(flat_logits, dim=-1)            # => (B*seq_len*D, C)
        per_token_logps = torch.gather(log_probs, dim=-1,
                                    index=flat_labels.unsqueeze(-1)).squeeze(-1)
        # => shape (B*seq_len*D,)

        # 6) Reshape back to (B, seq_len, D) so we can sum for each sample in the batch
        per_token_logps = per_token_logps.view(B, seq_len, D)
        loss_mask       = loss_mask.view(B, seq_len, D)

        # 7) Sum over seq_len, D => shape = (B,)
        logps_per_batch = (per_token_logps * loss_mask).sum(dim=(1, 2))
        valid_length    = loss_mask.sum(dim=(1, 2))

        return logps_per_batch, valid_length
