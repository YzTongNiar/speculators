# ruff: noqa: ERA001
import copy
import warnings
from typing import ClassVar

import torch
from torch.nn.attention.flex_attention import create_block_mask
from transformers import AutoConfig, DynamicCache, PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import SpeculatorModel
from speculators.models.eagle3 import Eagle3SpeculatorConfig, VwnEagle3SpeculatorConfig
from speculators.models.eagle3.attention import (
    create_combined_mask_mod,
    extend_mask_for_draft_tokens,
)
from speculators.models.eagle3.model_definitions import model_classes
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.utils.loading import load_model_layers

import os

def align_for_step(
    logits: torch.Tensor,  # shape: [1, total_seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, total_seq_len, draft_vocab_size]
    loss_mask: torch.Tensor | None,  # shape: [1, total_seq_len]
    prev_correct: torch.Tensor | None,  # shape: [1, total_seq_len]
    ttt_step: int,
):
    """Align logits, targets, loss_mask, and prev_correct for a given ttt_step.

    There are no target values for the last ttt_step tokens, so we mask them out
    before computing the loss/accuracy. Likewise, there are no logits for the first
    ttt_step tokens, so we mask them out.
    This is equivalent to shifting the target values by ttt_step + 1 to the left
    which puts them in the correct position for the generated tokens.
    e.g.
        indices of targets = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        indices of logits for ttt_step_0 = [1, 2, 3, 4, 5, 6, 7, 8, 9] # no shift
        indices of logits for ttt_step_1 = [2, 3, 4, 5, 6, 7, 8, 9, 10] # shift by 1
        indices of logits for ttt_step_2 = [3, 4, 5, 6, 7, 8, 9, 10, 11] # shift by 2
    The indices for the loss_mask need to be kept in line with the targets indices
    """
    logits = logits[:, :-ttt_step] if ttt_step > 0 else logits
    # shape: [1, total_seq_len - ttt_step, draft_vocab_size]
    targets = targets[:, ttt_step:]
    # shape: [1, total_seq_len - ttt_step, draft_vocab_size]
    if loss_mask is not None:
        loss_mask = loss_mask[:, ttt_step:]
        # shape: [1, total_seq_len - ttt_step]
    if prev_correct is not None:
        # Align with draft starts
        prev_correct = prev_correct[:, :-ttt_step] if ttt_step > 0 else prev_correct
        # shape: [1, total_seq_len - ttt_step]
    return logits, targets, loss_mask, prev_correct


@torch.no_grad()
def compute_accuracy(
    logits: torch.Tensor,  # shape: [1, total_seq_len - ttt_step, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, total_seq_len - ttt_step, draft_vocab_size]
    loss_mask: torch.Tensor | None,  # shape: [1, total_seq_len - ttt_step]
    prev_correct: torch.Tensor | None,  # shape: [1, total_seq_len - ttt_step]
):
    # Note: logits, targets, and loss_mask are already aligned for the current ttt_step
    target_tokens = torch.argmax(targets, dim=-1)
    predicted_tokens = torch.argmax(logits, dim=-1)
    # shape: [1, total_seq_len - ttt_step]

    correct = predicted_tokens == target_tokens
    cond_denom: torch.Tensor | int = correct.numel()
    if prev_correct is not None:
        cond_denom = prev_correct.sum()
        # Update prev_correct in place
        correct = torch.logical_and(prev_correct, correct, out=prev_correct)
    if loss_mask is not None:
        correct = torch.masked_select(correct, loss_mask.to(torch.bool))

    correct_sum = correct.float().sum()
    full_denom = correct.numel()

    return correct_sum / (full_denom + 1e-5), correct_sum / (cond_denom + 1e-5)


def loss_function(
    logits: torch.Tensor,  # shape: [1, total_seq_len - ttt_step, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, total_seq_len - ttt_step, draft_vocab_size]
    loss_mask: torch.Tensor | None,  # shape: [1, total_seq_len - ttt_step]
    loss_type: str = "kl",
    eta: float = 3.0,
) -> tuple[torch.Tensor, dict]:
    """Compute the training loss.

    Args:
        logits: Draft model logits (pre-softmax).
        targets: Target model logits (pre-softmax).
        loss_mask: Boolean mask over sequence positions.
        loss_type: One of "kl", "lk_log_acceptance", or "lk_hybrid".
            - "kl": Standard forward KL divergence KL(p||q).
            - "lk_log_acceptance": Negative log acceptance rate -log(alpha),
              where alpha = sum_x min(p(x), q(x)).  Eq. (L_LK^alpha) from the
              LK Losses paper (Samarin et al., 2026).
            - "lk_hybrid": Adaptive mixture lambda*KL + (1-lambda)*TV, with
              lambda = exp(-eta * sg[alpha]).  Eq. (L_LK^lambda) from the paper.
        eta: Decay rate for the adaptive lambda schedule (lk_hybrid only).

    Returns:
        Tuple of (scalar loss, extras dict). The extras dict contains:
            - "alpha": mean acceptance rate (lk_log_acceptance, lk_hybrid only)
            - "lambda": blending weight (lk_hybrid only)
    """
    # Note: logits, targets, and loss_mask are already aligned for the current ttt_step
    log_q = torch.nn.functional.log_softmax(logits, dim=-1)
    target_p = torch.nn.functional.softmax(targets, dim=-1)

    if loss_mask is not None:
        denominator: torch.Tensor | int = loss_mask.sum(dim=1) + 1e-5
    else:
        denominator = logits.shape[1]  # total_seq_len - ttt_step

    extras: dict = {}

    if loss_type == "kl":
        elementwise_loss = torch.nn.functional.kl_div(
            log_q, target_p, reduction="none", log_target=False
        )
        if loss_mask is not None:
            elementwise_loss = elementwise_loss * loss_mask.unsqueeze(-1)
        batch_loss = torch.sum(elementwise_loss, dim=(1, 2)) / denominator

    elif loss_type == "lk_log_acceptance":
        # L_LK^alpha = -log(alpha), alpha = sum_x min(p(x), q(x))
        # Gradient: (1/alpha) * grad(TV), which rescales TV gradient by 1/alpha,
        # restoring O(1/sqrt(k)) magnitude matching KL at initialisation.
        q = log_q.exp()
        alpha = torch.sum(torch.minimum(target_p, q), dim=-1)
        # shape: [1, total_seq_len - ttt_step]
        elementwise_loss = -torch.log(alpha + 1e-10)
        # shape: [1, total_seq_len - ttt_step]
        if loss_mask is not None:
            elementwise_loss = elementwise_loss * loss_mask
        batch_loss = torch.sum(elementwise_loss, dim=1) / denominator
        with torch.no_grad():
            if loss_mask is not None:
                extras["alpha"] = (alpha * loss_mask).sum() / (loss_mask.sum() + 1e-5)
            else:
                extras["alpha"] = alpha.mean()

    elif loss_type == "lk_hybrid":
        # L_LK^lambda = lambda * KL(p||q) + (1 - lambda) * TV(p, q)
        # lambda = exp(-eta * sg[alpha]), computed per draft-head position
        # (aggregated across batch and sequence dimensions).
        q = log_q.exp()
        with torch.no_grad():
            alpha = torch.sum(torch.minimum(target_p, q), dim=-1)
            # shape: [1, total_seq_len - ttt_step]
            if loss_mask is not None:
                alpha_mean = (alpha * loss_mask).sum() / (loss_mask.sum() + 1e-5)
            else:
                alpha_mean = alpha.mean()
            lambda_w = torch.exp(-eta * alpha_mean)
            extras["alpha"] = alpha_mean
            extras["lambda"] = lambda_w

        kl_per_pos = torch.sum(
            torch.nn.functional.kl_div(log_q, target_p, reduction="none", log_target=False),
            dim=-1,
        )  # shape: [1, total_seq_len - ttt_step]
        tv_per_pos = 0.5 * torch.sum(torch.abs(target_p - q), dim=-1)
        # shape: [1, total_seq_len - ttt_step]
        elementwise_loss = lambda_w * kl_per_pos + (1 - lambda_w) * tv_per_pos
        if loss_mask is not None:
            elementwise_loss = elementwise_loss * loss_mask
        batch_loss = torch.sum(elementwise_loss, dim=1) / denominator

    else:
        raise ValueError(
            f"Unknown loss_type: '{loss_type}'. "
            "Choose from 'kl', 'lk_log_acceptance', 'lk_hybrid'."
        )

    # shape: [1]
    return batch_loss.mean(), extras


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None,
    prev_correct: torch.Tensor | None,
    ttt_step: int,
    ttt_step_loss_decay: float,
    loss_type: str = "kl",
    eta: float = 3.0,
) -> tuple[torch.Tensor, dict]:
    """Compute metrics for a given ttt_step.

    Args:
        logits: The logits for the current ttt_step.
        targets: The targets for the current ttt_step.
        loss_mask: The loss mask for the current ttt_step.
        prev_correct: The previous correct predictions for the current ttt_step.
        ttt_step: The current ttt_step.
        ttt_step_loss_decay: The loss decay for the current ttt_step.
        loss_type: Loss variant — "kl", "lk_log_acceptance", or "lk_hybrid".
        eta: Decay rate for the adaptive lambda schedule (lk_hybrid only).

    Effects:
        Modifies prev_correct in place.

    Returns:
        Loss value and metrics dictionary.
    """

    s_metrics = {}
    s_logits, s_targets, s_loss_mask, s_prev_correct = align_for_step(
        logits, targets, loss_mask, prev_correct, ttt_step
    )
    loss_weight = ttt_step_loss_decay**ttt_step
    raw_loss, extras = loss_function(s_logits, s_targets, s_loss_mask, loss_type, eta)
    s_loss = loss_weight * raw_loss

    s_full_acc, s_cond_acc = compute_accuracy(
        s_logits, s_targets, s_loss_mask, s_prev_correct
    )
    s_metrics[f"loss_{ttt_step}"] = s_loss.detach().clone()
    s_metrics[f"full_acc_{ttt_step}"] = s_full_acc
    s_metrics[f"cond_acc_{ttt_step}"] = s_cond_acc
    for key, value in extras.items():
        s_metrics[f"{key}_{ttt_step}"] = value.detach().clone() if isinstance(value, torch.Tensor) else value

    return s_loss, s_metrics


def conditional_torch_compile(func):
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        return torch.compile(func)
    else:
        return func


@SpeculatorModel.register("eagle3")
class Eagle3DraftModel(SpeculatorModel):
    config_class: ClassVar[type[Eagle3SpeculatorConfig]] = Eagle3SpeculatorConfig  # type: ignore[misc]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "verifier_norm.weight",
        "verifier_lm_head.weight",
        "d2t",
        "t2d",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]

    def __init__(
        self,
        config: Eagle3SpeculatorConfig,
        t2d: torch.Tensor | None,
        d2t: torch.Tensor | None,
    ):
        super().__init__(
            config=config,
            verifier=None,
            verifier_attachment_mode="train_only",
        )
        self.hidden_size = config.transformer_layer_config.hidden_size
        self.draft_vocab_size = config.draft_vocab_size

        # Verify that if one mapping tensor is provided, the other is as well
        if (t2d is None) != (d2t is None):
            raise ValueError(
                "Both t2d and d2t must be provided together, or both must be None. "
                f"Got t2d={'provided' if t2d is not None else 'None'}, "
                f"d2t={'provided' if d2t is not None else 'None'}"
            )

        # Register buffers - they can be None
        if t2d is not None:
            self.register_buffer("t2d", t2d)  # shape: [verifier_vocab_size], bool
            if int(t2d.sum(dtype=torch.long).item()) != self.draft_vocab_size:
                raise ValueError(
                    f"t2d has {int(t2d.sum(dtype=torch.long).item())} non-zero values, "
                    f"expected {self.draft_vocab_size}."
                )
        else:
            self.register_buffer("t2d", None)

        if d2t is not None:
            self.register_buffer("d2t", d2t)  # shape: [draft_vocab_size], int offsets
            if d2t.shape[0] != self.draft_vocab_size:
                raise ValueError(
                    f"d2t.shape[0] ({d2t.shape[0]}) must match"
                    f" draft_vocab_size ({self.draft_vocab_size})."
                )
        else:
            self.register_buffer("d2t", None)

        self.fc = torch.nn.Linear(3 * self.hidden_size, self.hidden_size, bias=False)
        self._model_definitions = model_classes[
            config.transformer_layer_config.model_type
        ]
        self._setup_decoder_layers(
            config.transformer_layer_config, config.norm_before_residual
        )
        self.norm = self._model_definitions.norm_class(
            self.hidden_size, eps=config.transformer_layer_config.rms_norm_eps
        )
        self._setup_rotary_embedding(config.transformer_layer_config)
        self._setup_embeddings_and_lm_heads(
            config.speculators_config.verifier, t2d, config.embed_requires_grad
        )

    def _setup_decoder_layers(
        self, transformer_layer_config: PretrainedConfig, norm_before_residual: bool
    ):
        num_hidden_layers = transformer_layer_config.num_hidden_layers
        # Add first layer
        layers = [
            self._model_definitions.first_layer_class(
                transformer_layer_config,
                layer_idx=0,
                norm_before_residual=norm_before_residual,
            )
        ]
        # Add additional regular decoder layers
        layers.extend(
            [
                self._model_definitions.decoder_layer_class(
                    transformer_layer_config, layer_idx
                )
                for layer_idx in range(1, num_hidden_layers)
            ]
        )
        self.layers = torch.nn.ModuleList(layers)

    def _setup_rotary_embedding(self, transformer_layer_config: PretrainedConfig):
        # Create a modified config for the rotary embedding to use 2x the hidden size
        modified_config = copy.copy(transformer_layer_config)
        modified_config.hidden_size = modified_config.hidden_size * 2
        self.rotary_emb = self._model_definitions.rotary_emb_class(modified_config)

    def _setup_embeddings_and_lm_heads(
        self,
        config: VerifierConfig,
        t2d: torch.Tensor | None,
        embed_requires_grad: bool,
    ):
        if config.name_or_path is None:
            raise ValueError("VerifierConfig `name_or_path` value is required.")
        verifier_model_config = AutoConfig.from_pretrained(config.name_or_path)

        # For multimodal models (Qwen3VL, etc.), extract text_config
        if hasattr(verifier_model_config, "text_config"):
            verifier_model_config = verifier_model_config.text_config

        if verifier_model_config.hidden_size != self.hidden_size:
            raise ValueError(
                f"Verifier hidden size {verifier_model_config.hidden_size} does not"
                f" match draft hidden size {self.hidden_size}."
            )
        if t2d is not None and t2d.shape[0] != verifier_model_config.vocab_size:
            raise ValueError(
                f"t2d.shape[0] ({t2d.shape[0]}) must match"
                f" verifier_vocab_size ({verifier_model_config.vocab_size})."
            )

        # Load embedding and lm_head weights using suffix patterns (model-agnostic)
        verifier_weights = load_model_layers(
            ["embed_tokens.weight", "lm_head.weight", "model.norm.weight"],
            config.name_or_path,
        )

        if "embed_tokens.weight" not in verifier_weights:
            raise KeyError(
                f"Could not find embedding weights in {config.name_or_path}. "
                "Expected a key ending with 'embed_tokens.weight'."
            )

        embed_tokens_weight = verifier_weights["embed_tokens.weight"]
        # Use embed_tokens as fallback for lm_head if not found (tied weights)
        lm_head_weight = verifier_weights.get("lm_head.weight", embed_tokens_weight)
        self.verifier_norm = self._model_definitions.norm_class(
            self.hidden_size,
            eps=verifier_model_config.rms_norm_eps,
        )
        # EMBEDDINGS
        self.embed_tokens = torch.nn.Embedding(
            verifier_model_config.vocab_size,
            self.hidden_size,
            padding_idx=verifier_model_config.pad_token_id,
        )
        # shape: [verifier_vocab_size, hidden_size]
        default_dtype = self.embed_tokens.weight.dtype

        embed_tokens_sd = {"weight": embed_tokens_weight.to(default_dtype)}
        self.embed_tokens.load_state_dict(embed_tokens_sd)
        self.embed_tokens.weight.requires_grad = embed_requires_grad

        # LM HEADS
        self.lm_head = torch.nn.Linear(
            self.hidden_size, self.draft_vocab_size, bias=False
        )
        # shape: [hidden_size, draft_vocab_size]
        self.verifier_lm_head = torch.nn.Linear(
            self.hidden_size, self.draft_vocab_size, bias=False
        )

        if t2d is not None:
            # Reduce to limited vocab
            lm_head_weight = lm_head_weight.to(device=t2d.device, dtype=default_dtype)[
                t2d.to(torch.bool), :
            ]
        else:
            # Use full verifier vocab (no masking)
            lm_head_weight = lm_head_weight.to(dtype=default_dtype)
        if lm_head_weight.shape != self.lm_head.weight.shape:
            raise ValueError(
                f"Verifier lm head data shape "
                f"{lm_head_weight.shape} does not match draft "
                f"lm head shape {self.lm_head.weight.shape}"
            )
        self.lm_head.weight.data = lm_head_weight.detach().clone()
        self.verifier_lm_head.weight.data = lm_head_weight.detach().clone()

        self.verifier_lm_head.weight.requires_grad = False

        if "model.norm.weight" not in verifier_weights:
            warnings.warn(
                f"Could not find final norm weights in {config.name_or_path}. "
                "Using default initialization (weight=1.0).",
                UserWarning,
                stacklevel=2,
            )
        else:
            verifier_norm_weight = verifier_weights["model.norm.weight"]
            verifier_norm_sd = {"weight": verifier_norm_weight.to(default_dtype)}
            self.verifier_norm.load_state_dict(verifier_norm_sd)

        self.verifier_norm.weight.requires_grad = False

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [1, total_seq_len, 3 * hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        lengths: torch.Tensor | None = None,  # shape: [batch_size]
        loss_mask: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor
        | None = None,  # shape: [1, total_seq_len, hidden_size]
        ttt_steps: int = 3,
        ttt_step_loss_decay: float = 1.0,
        use_off_policy_tokens: bool = False,
        loss_type: str = "kl",
        eta: float = 3.0,
        **kwargs,
    ):
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]

        if lengths is None:
            lengths = torch.tensor([total_seq_len], dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = 1 + torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)
            # shape: [1, total_seq_len]

        past_key_values = DynamicCache(config=self.config.transformer_layer_config)

        combined_mask_mod = create_combined_mask_mod(lengths.to(device), total_seq_len)
        # Note: Attention mask is stored as a BlockMask object
        attention_mask = create_block_mask(
            combined_mask_mod,
            B=None,
            H=None,
            Q_LEN=total_seq_len,
            KV_LEN=total_seq_len,
            device=device,
        )

        hidden_states = self.fc(hidden_states)
        # shape: [1, total_seq_len, hidden_size]

        original_input_ids = input_ids.detach().clone()
        return_loss = verifier_last_hidden_states is not None
        if return_loss:
            with torch.no_grad():
                targets = self.verifier_lm_head(
                    self.verifier_norm(verifier_last_hidden_states)
                )
                # shape: [1, total_seq_len, draft_vocab_size]
            loss = torch.tensor(0.0, device=device)

            # prev_correct is a boolean tensor that is True for tokens that have been
            # correctly predicted on all previous ttt_steps.
            # Initialized to True if the token is included in the loss_mask
            # or if there is no loss_mask
            prev_correct = (
                loss_mask.clone()
                if loss_mask is not None
                else torch.ones(1, total_seq_len, device=device, dtype=torch.bool)
            )
            metrics = {}

        draft_tokens = []
        for ttt_step in range(ttt_steps):
            with torch.no_grad():
                input_embeds = self.embed_tokens(input_ids)
                # shape: [1, total_seq_len, hidden_size]
            cache_position = torch.arange(
                ttt_step * total_seq_len,
                (ttt_step + 1) * total_seq_len,
                dtype=torch.long,
                device=device,
            )
            # shape: [total_seq_len]

            hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)
            # shape: [1, total_seq_len, 2 * hidden_size]

            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            for decoder_layer in self.layers:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            logits = self.lm_head(self.norm(hidden_states))
            # shape: [1, total_seq_len, draft_vocab_size]

            if return_loss:
                s_loss, s_metrics = compute_metrics(
                    logits,
                    targets,
                    loss_mask,
                    prev_correct,
                    ttt_step,
                    ttt_step_loss_decay,
                    loss_type,
                    eta,
                )
                loss += s_loss
                metrics.update(s_metrics)

            input_ids = torch.argmax(logits, dim=-1)
            draft_tokens.append(input_ids.detach().clone())
            # shape: [1, total_seq_len]
            # Use d2t to map draft tokens to verifier tokens.
            # Must be in verifier vocabulary space because we use the full verifier
            # vocabulary in the embedding.
            if self.d2t is not None:
                input_ids = input_ids + self.d2t[input_ids]  # type: ignore[index]

            if use_off_policy_tokens:
                # Overwrite input_ids with ground truth tokens
                # shift input_ids by 1 to the left and pad with 0
                # note: inputs_ids no longer line up with verifier_last_hidden_states
                # the draft logits generated from the padded tokens are ignored
                # and sliced out for loss calculation
                input_ids = torch.cat(
                    [
                        original_input_ids[:, 1 + ttt_step :],
                        original_input_ids.new_zeros(1, 1 + ttt_step),
                    ],
                    dim=-1,
                )
                # shape: [1, total_seq_len]

            attention_mask = extend_mask_for_draft_tokens(attention_mask)
            position_ids = position_ids + 1
            # shape: [1, total_seq_len]

        if return_loss:
            metrics["loss"] = loss.detach().clone()
            return draft_tokens, loss, metrics
        else:
            return draft_tokens

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        **kwargs,
    ) -> "Eagle3DraftModel":
        """Create Eagle3 model from training arguments.

        Args:
            verifier_config: Verifier model configuration
            **kwargs: Training arguments with Eagle3-specific params
                - num_layers: Number of decoder layers
                - norm_before_residual: Whether to normalize before residual connection
                - t2d: Target-to-draft vocabulary mapping tensor
                - d2t: Draft-to-target vocabulary mapping tensor
                - ttt_steps: Number of TTT steps
                - verifier_name_or_path: Path to verifier model

        Returns:
            Initialized Eagle3DraftModel
        """
        config = Eagle3SpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            norm_before_residual=kwargs["norm_before_residual"],
            embed_requires_grad=kwargs.get("embed_requires_grad", False),
            speculators_config=SpeculatorsConfig(
                algorithm="eagle3",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        speculative_tokens=kwargs["ttt_steps"],
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(
                    verifier_config, name_or_path=kwargs["verifier_name_or_path"]
                ),
            ),
        )

        return cls(config=config, t2d=kwargs.get("t2d"), d2t=kwargs.get("d2t"))

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Get training and validation kwargs for Eagle3.

        Args:
            **kwargs: Training arguments

        Returns:
            Tuple of (train_call_kwargs, val_call_kwargs)
        """
        train_kwargs = {
            "use_off_policy_tokens": kwargs["use_off_policy_tokens"],
            "ttt_steps": kwargs["ttt_steps"],
            "ttt_step_loss_decay": kwargs["ttt_step_loss_decay"],
            "loss_type": kwargs.get("loss_type", "kl"),
            "eta": kwargs.get("eta", 3.0),
        }
        val_kwargs = {
            "use_off_policy_tokens": False,
            "ttt_steps": kwargs["ttt_steps"],
            "ttt_step_loss_decay": kwargs["ttt_step_loss_decay"],
            "loss_type": kwargs.get("loss_type", "kl"),
            "eta": kwargs.get("eta", 3.0),
        }
        return train_kwargs, val_kwargs

# [改动2] 注册为 "vwn_eagle3"
@SpeculatorModel.register("vwn_eagle3")
class VwnEagle3DraftModel(SpeculatorModel):
    # [改动3] config_class 指向 VwnEagle3SpeculatorConfig
    config_class: ClassVar[type[VwnEagle3SpeculatorConfig]] = VwnEagle3SpeculatorConfig  # type: ignore[misc]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "verifier_norm.weight",
        "verifier_lm_head.weight",
        "d2t",
        "t2d",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]

    count = 0

    # ──────────────────────────────────────────────────────────────────────
    # __init__（与 Eagle3DraftModel.__init__ 基本相同）
    # ──────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        config: VwnEagle3SpeculatorConfig,   # [改动4a] 类型更新
        t2d: torch.Tensor | None,
        d2t: torch.Tensor | None,
    ):
        super().__init__(
            config=config,
            verifier=None,
            verifier_attachment_mode="train_only",
        )
        self.hidden_size = config.transformer_layer_config.hidden_size
        self.draft_vocab_size = config.draft_vocab_size

        # [改动4b] 存储 VwnConfig，供 _setup_decoder_layers 使用
        self.vwn = config.vwn
        self.pre_vwn_layer_class = config.pre_vwn_layer_class

        # ── 以下与 Eagle3DraftModel 完全相同 ─────────────────────────────
        if (t2d is None) != (d2t is None):
            raise ValueError(
                "Both t2d and d2t must be provided together, or both must be None. "
                f"Got t2d={'provided' if t2d is not None else 'None'}, "
                f"d2t={'provided' if d2t is not None else 'None'}"
            )

        if t2d is not None:
            self.register_buffer("t2d", t2d)
            if int(t2d.sum(dtype=torch.long).item()) != self.draft_vocab_size:
                raise ValueError(
                    f"t2d has {int(t2d.sum(dtype=torch.long).item())} non-zero values, "
                    f"expected {self.draft_vocab_size}."
                )
        else:
            self.register_buffer("t2d", None)

        if d2t is not None:
            self.register_buffer("d2t", d2t)
            if d2t.shape[0] != self.draft_vocab_size:
                raise ValueError(
                    f"d2t.shape[0] ({d2t.shape[0]}) must match"
                    f" draft_vocab_size ({self.draft_vocab_size})."
                )
        else:
            self.register_buffer("d2t", None)

        self.fc = torch.nn.Linear(3 * self.hidden_size, self.hidden_size, bias=False)
        self._model_definitions = model_classes[
            config.vwn_draft_arch
        ]
        self._setup_decoder_layers(
            config.transformer_layer_config, config.norm_before_residual
        )
        self.norm = self._model_definitions.norm_class(
            self.hidden_size, eps=config.transformer_layer_config.rms_norm_eps
        )
        self._setup_rotary_embedding(config.transformer_layer_config)
        self._setup_embeddings_and_lm_heads(
            config.speculators_config.verifier, t2d, config.embed_requires_grad
        )
        self._logit_save_counter = 0

    def save_tensor_to_dir(self, tensor_data, sub_dir_name):
        import os
        save_path = "/mnt/share/t00886357/eagle3/qwen3_30b_gsm8k_fix_pattern/logits/vwn_eagle3/hidden"
        file_name_new = f"{sub_dir_name}_{self.count}.pth"
        target_dir = os.path.join(save_path, sub_dir_name)
        os.makedirs(target_dir, exist_ok=True)
        save_path = os.path.join(target_dir, file_name_new)
        print(f"\n\n layer save tensors {sub_dir_name}, shape is {tensor_data.shape}, file_name is {file_name_new}")
        self.count += 1

    # ──────────────────────────────────────────────────────────────────────
    # [改动5] _setup_decoder_layers：first_layer_class 需要额外接收 vwn 参数
    # ──────────────────────────────────────────────────────────────────────
    def _setup_decoder_layers(
        self, transformer_layer_config: PretrainedConfig, norm_before_residual: bool
    ):
        num_hidden_layers = transformer_layer_config.num_hidden_layers
        layers = [
            self._model_definitions.first_layer_class(
                transformer_layer_config,
                layer_idx=0,
                vwn=self.vwn,                          # [改动5] 传入 vwn
                norm_before_residual=norm_before_residual,
                pre_vwn_layer_class=self.pre_vwn_layer_class
            )
        ]
        layers.extend(
            [
                self._model_definitions.decoder_layer_class(
                    transformer_layer_config, layer_idx
                )
                for layer_idx in range(1, num_hidden_layers)
            ]
        )
        self.layers = torch.nn.ModuleList(layers)

    # ── 以下两个方法与 Eagle3DraftModel 完全相同，原样复制 ─────────────────

    def _setup_rotary_embedding(self, transformer_layer_config: PretrainedConfig):
        # 旋转嵌入仍然基于 2×hidden_size（第一层输入仍是 cat([embeds, hidden])）
        modified_config = copy.copy(transformer_layer_config)
        modified_config.hidden_size = modified_config.hidden_size * 2
        self.rotary_emb = self._model_definitions.rotary_emb_class(modified_config)

    def _setup_embeddings_and_lm_heads(
        self,
        config: VerifierConfig,
        t2d: torch.Tensor | None,
        embed_requires_grad: bool,
    ):
        if config.name_or_path is None:
            raise ValueError("VerifierConfig `name_or_path` value is required.")
        verifier_model_config = AutoConfig.from_pretrained(config.name_or_path)

        if hasattr(verifier_model_config, "text_config"):
            verifier_model_config = verifier_model_config.text_config

        if verifier_model_config.hidden_size != self.hidden_size:
            raise ValueError(
                f"Verifier hidden size {verifier_model_config.hidden_size} does not"
                f" match draft hidden size {self.hidden_size}."
            )
        if t2d is not None and t2d.shape[0] != verifier_model_config.vocab_size:
            raise ValueError(
                f"t2d.shape[0] ({t2d.shape[0]}) must match"
                f" verifier_vocab_size ({verifier_model_config.vocab_size})."
            )

        verifier_weights = load_model_layers(
            ["embed_tokens.weight", "lm_head.weight", "model.norm.weight"],
            config.name_or_path,
        )

        if "embed_tokens.weight" not in verifier_weights:
            raise KeyError(
                f"Could not find embedding weights in {config.name_or_path}. "
                "Expected a key ending with 'embed_tokens.weight'."
            )

        embed_tokens_weight = verifier_weights["embed_tokens.weight"]
        lm_head_weight = verifier_weights.get("lm_head.weight", embed_tokens_weight)
        self.verifier_norm = self._model_definitions.norm_class(
            self.hidden_size,
            eps=verifier_model_config.rms_norm_eps,
        )
        self.embed_tokens = torch.nn.Embedding(
            verifier_model_config.vocab_size,
            self.hidden_size,
            padding_idx=verifier_model_config.pad_token_id,
        )
        default_dtype = self.embed_tokens.weight.dtype

        self.embed_tokens.load_state_dict(
            {"weight": embed_tokens_weight.to(default_dtype)}
        )
        self.embed_tokens.weight.requires_grad = embed_requires_grad

        self.lm_head = torch.nn.Linear(
            self.hidden_size, self.draft_vocab_size, bias=False
        )
        self.verifier_lm_head = torch.nn.Linear(
            self.hidden_size, self.draft_vocab_size, bias=False
        )

        if t2d is not None:
            lm_head_weight = lm_head_weight.to(device=t2d.device, dtype=default_dtype)[
                t2d.to(torch.bool), :
            ]
        else:
            lm_head_weight = lm_head_weight.to(dtype=default_dtype)
        if lm_head_weight.shape != self.lm_head.weight.shape:
            raise ValueError(
                f"Verifier lm head data shape "
                f"{lm_head_weight.shape} does not match draft "
                f"lm head shape {self.lm_head.weight.shape}"
            )
        self.lm_head.weight.data = lm_head_weight.detach().clone()
        self.verifier_lm_head.weight.data = lm_head_weight.detach().clone()
        self.verifier_lm_head.weight.requires_grad = False

        if "model.norm.weight" not in verifier_weights:
            warnings.warn(
                f"Could not find final norm weights in {config.name_or_path}. "
                "Using default initialization (weight=1.0).",
                UserWarning,
                stacklevel=2,
            )
        else:
            verifier_norm_weight = verifier_weights["model.norm.weight"]
            self.verifier_norm.load_state_dict(
                {"weight": verifier_norm_weight.to(default_dtype)}
            )

        self.verifier_norm.weight.requires_grad = False

    # ──────────────────────────────────────────────────────────────────────
    # [改动6] forward：TTT 循环中拆分 first_layer / rest_layers 调用
    # ──────────────────────────────────────────────────────────────────────
    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [1, total_seq_len, 3 * hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        lengths: torch.Tensor | None = None,  # shape: [batch_size]
        loss_mask: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor
        | None = None,  # shape: [1, total_seq_len, hidden_size]
        ttt_steps: int = 3,
        ttt_step_loss_decay: float = 1.0,
        use_off_policy_tokens: bool = False,
        loss_type: str = "kl",
        eta: float = 3.0,
        **kwargs,
    ):

        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]

        if lengths is None:
            lengths = torch.tensor([total_seq_len], dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = 1 + torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        past_key_values = DynamicCache(config=self.config.transformer_layer_config)

        combined_mask_mod = create_combined_mask_mod(lengths.to(device), total_seq_len)
        attention_mask = create_block_mask(
            combined_mask_mod,
            B=None,
            H=None,
            Q_LEN=total_seq_len,
            KV_LEN=total_seq_len,
            device=device,
        )

        # FC 压缩：3D → D（与 Eagle3 相同）
        hidden_states = self.fc(hidden_states)
        # shape: [1, total_seq_len, hidden_size]

        original_input_ids = input_ids.detach().clone()
        return_loss = verifier_last_hidden_states is not None
        if return_loss:
            with torch.no_grad():
                targets = self.verifier_lm_head(
                    self.verifier_norm(verifier_last_hidden_states)
                )
            loss = torch.tensor(0.0, device=device)
            prev_correct = (
                loss_mask.clone()
                if loss_mask is not None
                else torch.ones(1, total_seq_len, device=device, dtype=torch.bool)
            )
            metrics = {}

        draft_tokens = []
        for ttt_step in range(ttt_steps):
            with torch.no_grad():
                input_embeds = self.embed_tokens(input_ids)
                # shape: [1, total_seq_len, hidden_size]
            cache_position = torch.arange(
                ttt_step * total_seq_len,
                (ttt_step + 1) * total_seq_len,
                dtype=torch.long,
                device=device,
            )

            # [改动6a] 构造 first_layer 输入：仍是 cat([embeds, hidden])
            # Eagle3 在这里直接 hidden_states = cat(...)，然后所有层都接受 2D 输入。
            # VWN 的 first_layer（VwnEagle3FirstLayerMixin）接受 2D 输入，
            # 内部完成 pre_vwn 融合后输出 D，后续层接受 D 输入，故需要分开调用。
            first_layer_input = torch.cat([input_embeds, hidden_states], dim=-1)
            # shape: [1, total_seq_len, 2 * hidden_size]

            # rotary embedding 基于 2×D 空间（与 Eagle3 相同）
            position_embeddings = self.rotary_emb(first_layer_input, position_ids)

            ## TODO: pre_vwn
            # [改动6b] 第一层：VWN first layer，输入 2D，输出 D
            hidden_states = self.layers[0](
                first_layer_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            # shape: [1, total_seq_len, hidden_size]

            # [改动6c] 后续标准 decoder layers（若 num_hidden_layers > 1）
            for decoder_layer in self.layers[1:]:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            # shape: [1, total_seq_len, hidden_size]

            logits = self.lm_head(self.norm(hidden_states))
            # shape: [1, total_seq_len, draft_vocab_size]

            if return_loss:
                s_loss, s_metrics = compute_metrics(
                    logits,
                    targets,
                    loss_mask,
                    prev_correct,
                    ttt_step,
                    ttt_step_loss_decay,
                    loss_type,
                    eta,
                )
                loss += s_loss
                metrics.update(s_metrics)

            input_ids = torch.argmax(logits, dim=-1)
            draft_tokens.append(input_ids.detach().clone())

            if self.d2t is not None:
                input_ids = input_ids + self.d2t[input_ids]  # type: ignore[index]

            if use_off_policy_tokens:
                input_ids = torch.cat(
                    [
                        original_input_ids[:, 1 + ttt_step:],
                        original_input_ids.new_zeros(1, 1 + ttt_step),
                    ],
                    dim=-1,
                )

            attention_mask = extend_mask_for_draft_tokens(attention_mask)
            position_ids = position_ids + 1

        if return_loss:
            metrics["loss"] = loss.detach().clone()
            return draft_tokens, loss, metrics
        else:
            return draft_tokens

    # ──────────────────────────────────────────────────────────────────────
    # [改动7] from_training_args：config 类型改为 VwnEagle3SpeculatorConfig
    #                              并额外读取 vwn_m / vwn_r
    # ──────────────────────────────────────────────────────────────────────
    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        **kwargs,
    ) -> "VwnEagle3DraftModel":
        config = VwnEagle3SpeculatorConfig(          # [改动7a]
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            norm_before_residual=kwargs["norm_before_residual"],
            embed_requires_grad=kwargs.get("embed_requires_grad", False),
            vwn_draft_arch=kwargs.get('vwn_draft_arch', 'vwn_llama'),  # "vwn_llama" 或 "vwn_qwen3"
            vwn_m=kwargs.get("vwn_m", 2),            # [改动7b]
            vwn_r=kwargs.get("vwn_r", 1.5),          # [改动7c]
            pre_vwn_version=kwargs.get("pre_vwn_version", 0),  # [改动7d]
            speculators_config=SpeculatorsConfig(
                algorithm="vwn_eagle3",              # [改动7e]
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        speculative_tokens=kwargs["ttt_steps"],
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(
                    verifier_config, name_or_path=kwargs["verifier_name_or_path"]
                ),
            ),
        )
        return cls(config=config, t2d=kwargs.get("t2d"), d2t=kwargs.get("d2t"))

    # ── get_trainer_kwargs 与 Eagle3DraftModel 完全相同，原样复制 ──────────
    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        train_kwargs = {
            "use_off_policy_tokens": kwargs["use_off_policy_tokens"],
            "ttt_steps": kwargs["ttt_steps"],
            "ttt_step_loss_decay": kwargs["ttt_step_loss_decay"],
            "loss_type": kwargs.get("loss_type", "kl"),
            "eta": kwargs.get("eta", 3.0),
        }
        val_kwargs = {
            "use_off_policy_tokens": False,
            "ttt_steps": kwargs["ttt_steps"],
            "ttt_step_loss_decay": kwargs["ttt_step_loss_decay"],
            "loss_type": kwargs.get("loss_type", "kl"),
            "eta": kwargs.get("eta", 3.0),
        }
        return train_kwargs, val_kwargs