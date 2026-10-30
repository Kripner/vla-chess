#!/usr/bin/env python3
"""π0.5's unified block-causal sequence with stage prediction:

  [ images | whole-move prompt ]  [ subtask tokens ]  [ action expert tokens ]
     bidirectional prefix           causal, CE loss     flow loss, attends to all before

One forward, both losses: the model predicts the current sub-task text from the
observation + the whole-move prompt (CE on the subtask tokens through PaliGemma's LM
head), and the action expert attends to those tokens in context — lerobot's pi05 port
already builds block-causal attention from per-token flags, so this subclass only adds
the middle segment. Installed onto a stock PI05Policy by `install(policy)` (class swap;
the port stays untouched library code).

Batch contract: the trainer resolves each frame's sub-task NAME from the dataset's
language_persistent mark rows (chess.real.segments.name_of) and attaches it as a
SINGLE-TOKEN label (label_tokens/attach_subtask) under lerobot's subtask keys.

Why one token, not the full instruction sentence: a zero-shot switcher bench showed summed-logprob margins over the five sentences are dominated by static
text/length biases (per-token normalization reshuffled the flat offsets — no
observation signal either way), and the frozen head has no transition signal at all,
so the predictor must be finetuned (unfreeze_vlm_top) — which frees the label from
having to be natural language. Each sub-task name (approach/pick/transport/place/home)
is exactly one PaliGemma vocab token, so: the switcher margin is a single logit
difference, comparable across all four boundaries under one threshold; the CE is a
plain classification at the last prefix position; and the CE stream through the joint
attention is as small as it can be (see the stability note in forward_with_subtask).
The datasets' mark rows keep their full sentences as human-readable annotation; the
whole-move prompt in the prefix still carries source/target, so no information is lost.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from lerobot.policies.pi05.modeling_pi05 import (
    PI05Policy,
    PI05Pytorch,
    make_att_2d_masks,
    prepare_attention_masks_4d,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
)

LAMBDA_CE = 0.1     # subtask-CE weight in the total loss (a constant, not a flag)


def _last_real(pad_masks: torch.Tensor) -> torch.Tensor:
    """Per-sample index of the LAST real (unpadded) prefix position. The pipeline's
    tokenizer LEFT-pads the prompt, so `pad.sum()-1` points at a padding row — a row
    the 2D mask fully masks, which under the finite openpi mask constant attends the
    whole joint sequence UNIFORMLY (softmax of an all-equal row). Reading the CE there
    made it a garbage readout that leaked gradient into the expert's K/V and diverged
    every CE-attached run."""
    idx = torch.arange(pad_masks.shape[1], device=pad_masks.device)
    return (idx * pad_masks.long()).amax(dim=1)


class SubtaskPI05Pytorch(PI05Pytorch):
    def forward_with_subtask(self, images, img_masks, tokens, masks,
                             sub_tokens, sub_masks, actions, noise, time):
        """Stock training forward with the causal subtask segment between prefix and
        expert -> (flow losses as stock forward, subtask CE scalar)."""
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, tokens, masks)
        sub_embs = self.paligemma_with_expert.embed_language_tokens(sub_tokens)
        bsize, s_len = sub_tokens.shape
        # att flag 1 = "starts a new causal block": one per subtask token = strictly causal
        # within the segment; the expert (whose first token also starts a block, with a higher
        # cumulative index) attends to prefix + segment; the segment never sees the expert.
        sub_att = torch.ones(bsize, s_len, dtype=torch.bool, device=prefix_pad.device)
        suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(x_t, time)

        if (self.paligemma_with_expert.paligemma.model.language_model.layers[0]
                .self_attn.q_proj.weight.dtype == torch.bfloat16):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            sub_embs = sub_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        pg_embs = torch.cat([prefix_embs, sub_embs], dim=1)
        pad_masks = torch.cat([prefix_pad, sub_masks, suffix_pad], dim=1)
        att_masks = torch.cat([prefix_att, sub_att, suffix_att], dim=1)

        att_2d = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_4d = prepare_attention_masks_4d(att_2d)

        def forward_func(pg_embs, suffix_embs, att_4d, position_ids, adarms_cond):
            (pg_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[pg_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return pg_out, suffix_out

        pg_out, suffix_out = self._apply_checkpoint(
            forward_func, pg_embs, suffix_embs, att_4d, position_ids, adarms_cond)

        # -- flow loss (stock tail) --------------------------------------------------- #
        suffix_out = suffix_out[:, -self.config.chunk_size:].to(dtype=torch.float32)
        v_t = self._apply_checkpoint(self.action_out_proj, suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")

        # -- subtask CE (teacher-forced next-token over the segment) ------------------- #
        p_len = prefix_embs.shape[1]
        # position predicting subtask token 0 = the last REAL prefix token (_last_real:
        # the prompt is LEFT-padded, so sum()-1 would hit a garbage padding row).
        last_real = _last_real(prefix_pad)                           # (B,)
        hid = torch.cat([
            pg_out[torch.arange(bsize, device=pg_out.device), last_real][:, None],
            pg_out[:, p_len:p_len + s_len - 1],
        ], dim=1)                                                    # (B, S, D)
        # The readout must be the last REAL prefix position (see _last_real): pad.sum()-1 on
        # the LEFT-padded prompt is a fully-masked row attending the whole sequence uniformly,
        # which leaks an unlearnable ~40x-the-flow gradient into the expert's K/V and
        # diverges training. With train_expert_only the whole pg tower is frozen, so a (now clean) CE
        # still cannot train anything: detached it is a pure monitor. unfreeze_vlm_top
        # sets subtask_ce_attached so prediction actually trains.
        if (self.paligemma_with_expert.train_expert_only
                and not getattr(self, "subtask_ce_attached", False)):
            hid = hid.detach()
        logits = self.paligemma_with_expert.paligemma.lm_head(hid).to(torch.float32)
        ce = F.cross_entropy(logits.reshape(bsize * s_len, -1),
                             sub_tokens.reshape(-1),
                             reduction="none").reshape(bsize, s_len)
        ce = (ce * sub_masks).sum() / sub_masks.sum().clamp(min=1)
        with torch.no_grad():
            # accuracy over the label candidate set (subtask_label_ids, set by the
            # trainer) — the per-frame proxy of the deployed switcher, which compares
            # within {current, next}; without the ids, full-vocab argmax (strict bound)
            ids = getattr(self, "subtask_label_ids", None)
            if ids is not None:
                ids = ids.to(logits.device)
                pred = ids[logits[..., ids].argmax(-1)]
            else:
                pred = logits.argmax(-1)
            hits = ((pred == sub_tokens) & sub_masks).sum()
            self.last_subtask_acc = float(hits) / float(sub_masks.sum().clamp(min=1))
        return losses, ce


    # ---- inference ------------------------------------------------------------------ #
    @torch.no_grad()
    def label_logprobs(self, images, img_masks, tokens, masks, label_ids):
        """Log-probability of each single-token label as the first subtask token, given the
        prefix -> (B, n). label_ids: (n,) shared by every sample, or (B, n) per sample. ONE
        prefix-only forward per sample: the prefix cannot attend a later block (block-causal
        mask), so the logits at the last real prompt position are the same whichever label
        follows — no per-candidate repetition."""
        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, tokens, masks)
        bsize = prefix_pad.shape[0]
        if (self.paligemma_with_expert.paligemma.model.language_model.layers[0]
                .self_attn.q_proj.weight.dtype == torch.bfloat16):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        att_4d = prepare_attention_masks_4d(make_att_2d_masks(prefix_pad, prefix_att))
        position_ids = torch.cumsum(prefix_pad, dim=1) - 1
        # same eager switch as stock sample_actions (SDPA rejects the float mask against
        # bf16 queries on this path)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        (pg_out, _), _ = self.paligemma_with_expert.forward(
            attention_mask=att_4d, position_ids=position_ids, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=False)
        hid = pg_out[torch.arange(bsize, device=pg_out.device), _last_real(prefix_pad)]  # (B, D)
        logp = torch.log_softmax(
            self.paligemma_with_expert.paligemma.lm_head(hid).to(torch.float32), dim=-1)
        ids = label_ids.to(logp.device)
        if ids.dim() == 1:
            ids = ids[None].expand(bsize, -1)
        return logp.gather(-1, ids)                                    # (B, n)

    @torch.no_grad()
    def sample_actions_with_subtask(self, images, img_masks, tokens, masks,
                                    sub_tokens, sub_masks, noise=None, num_steps=None,
                                    **kwargs):
        """Stock sample_actions with the chosen subtask segment in the cached context —
        the expert attends to it exactly as in training."""
        from lerobot.policies.pi05.modeling_pi05 import euler_integrate
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        bsize = tokens.shape[0]
        device = tokens.device
        if noise is None:
            noise = self.sample_noise(
                (bsize, self.config.chunk_size, self.config.max_action_dim), device)

        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, tokens, masks)
        sub_embs = self.paligemma_with_expert.embed_language_tokens(sub_tokens)
        s_len = sub_tokens.shape[1]
        sub_att = torch.ones(bsize, s_len, dtype=torch.bool, device=prefix_pad.device)
        if (self.paligemma_with_expert.paligemma.model.language_model.layers[0]
                .self_attn.q_proj.weight.dtype == torch.bfloat16):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            sub_embs = sub_embs.to(dtype=torch.bfloat16)
        pg_embs = torch.cat([prefix_embs, sub_embs], dim=1)
        pad_masks = torch.cat([prefix_pad, sub_masks], dim=1)
        att_masks = torch.cat([prefix_att, sub_att], dim=1)
        att_4d = prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=att_4d, position_ids=position_ids, past_key_values=None,
            inputs_embeds=[pg_embs, None], use_cache=True)
        return euler_integrate(
            lambda x_t, t: self.denoise_step(
                prefix_pad_masks=pad_masks, past_key_values=past_key_values,
                x_t=x_t, timestep=t),
            noise, num_steps,
            rtc_processor=self.rtc_processor, rtc_enabled=self._rtc_enabled(),
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            execution_horizon=kwargs.get("execution_horizon"),
        )


class SubtaskPI05Policy(PI05Policy):
    def forward(self, batch: dict, reduction: str = "mean"):
        if OBS_LANGUAGE_SUBTASK_TOKENS not in batch:
            raise ValueError(
                "pi05 trains on the unified sub-task sequence: the batch has no "
                f"{OBS_LANGUAGE_SUBTASK_TOKENS} — record marked data (chess.real.segments) "
                "and train through chess.train.finetune_real.")
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        sub_tokens = batch[OBS_LANGUAGE_SUBTASK_TOKENS]
        sub_masks = batch[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)

        losses, ce = self.model.forward_with_subtask(
            images, img_masks, tokens, masks, sub_tokens, sub_masks, actions, noise, time)
        losses = losses[:, :, :self.config.output_features[ACTION].shape[0]]
        self.last_subtask_ce = float(ce.detach())
        self.last_subtask_acc = getattr(self.model, "last_subtask_acc", None)
        loss_dict = {"loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
                     "subtask_ce": self.last_subtask_ce,
                     "subtask_acc": self.last_subtask_acc}
        if reduction == "none":
            per_sample = losses.mean(dim=(1, 2)) + LAMBDA_CE * ce   # ce is batch-mean; per-sample
            loss_dict["loss"] = per_sample.mean().item()            # weighting keeps it shared
            return per_sample, loss_dict
        loss = losses.mean() + LAMBDA_CE * ce
        loss_dict["loss"] = loss.item()
        return loss, loss_dict


    def predict_action_chunk(self, batch: dict, **kwargs):
        """Stock predict, with the chosen subtask segment in context when the batch carries
        lerobot's subtask token keys (the switcher put them there)."""
        if OBS_LANGUAGE_SUBTASK_TOKENS not in batch:
            return super().predict_action_chunk(batch, **kwargs)
        self.eval()
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.model.sample_actions_with_subtask(
            images, img_masks, tokens, masks,
            batch[OBS_LANGUAGE_SUBTASK_TOKENS], batch[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK],
            **kwargs)
        return actions[:, :, :self.config.output_features[ACTION].shape[0]]

    @torch.no_grad()
    def score_subtasks(self, batch: dict, cand_tokens, cand_masks, n_cands: int = None):
        """Log-probability of single-token label candidates -> (B, n). Shared candidates:
        cand_tokens (n, 1) for every sample. Per-sample candidates: cand_tokens (B*n, 1)
        sample-major with n_cands=n. One prefix forward per sample (label_logprobs)."""
        self.eval()
        assert cand_tokens.shape[1] == 1, "sub-task labels are single tokens (label_tokens)"
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        b = tokens.shape[0]
        ids = cand_tokens[:, 0]
        if n_cands is not None:
            assert ids.shape[0] == b * n_cands, "per-sample candidates must be (B*n, 1)"
            ids = ids.reshape(b, n_cands)
        return self.model.label_logprobs(images, img_masks, tokens, masks, ids)


def install(policy) -> bool:
    """Swap a stock PI05Policy (and its inner model) to the subtask classes in place.
    Returns True when installed; no-op (False) for other families."""
    if isinstance(policy, PI05Policy) and not isinstance(policy, SubtaskPI05Policy):
        policy.__class__ = SubtaskPI05Policy
        policy.model.__class__ = SubtaskPI05Pytorch
        return True
    return isinstance(policy, SubtaskPI05Policy)


def find_tokenizer(pre):
    """The tokenizer instance of the pipeline's TokenizerProcessorStep (the subtask segment
    must be tokenized exactly like the prompt). The loaded instance lives in
    `input_tokenizer`; the `tokenizer` field is only set when one was passed explicitly."""
    for step in getattr(pre, "steps", []):
        tok = getattr(step, "input_tokenizer", None) or getattr(step, "tokenizer", None)
        if tok is not None:
            return tok
    raise RuntimeError("no tokenizer step in the preprocessor pipeline")


def label_tokens(tokenizer, names) -> tuple[torch.Tensor, torch.Tensor]:
    """((N,1) int64 ids, (N,1) bool masks) for sub-task labels: each NAME is exactly one
    vocab token (asserted) — the module docstring says why single tokens."""
    ids = []
    for n in names:
        t = tokenizer(n, add_special_tokens=False)["input_ids"]
        if len(t) != 1:
            raise ValueError(f"sub-task label {n!r} is not a single token: {t}")
        ids.append(t[0])
    tok = torch.tensor(ids, dtype=torch.long).unsqueeze(1)
    return tok, torch.ones_like(tok, dtype=torch.bool)


def attach_subtask(batch: dict, names: list[str], tokenizer) -> dict:
    """Attach the per-sample sub-task labels (names) under lerobot's subtask keys."""
    tok, masks = label_tokens(tokenizer, names)
    dev = batch[OBS_LANGUAGE_TOKENS].device
    batch[OBS_LANGUAGE_SUBTASK_TOKENS] = tok.to(dev)
    batch[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK] = masks.to(dev)
    return batch


def unfreeze_vlm_top(policy, n_layers: int) -> int:
    """Train the stage predictor: unfreeze the top
    `n_layers` LM layers + the final norm of the otherwise-frozen VLM and attach the
    subtask CE, so prediction actually trains (both CE and flow gradients then reach
    those layers through the shared joint attention). The lm_head — weight-tied to the
    input embeddings, so training it would move every prompt/label embedding — stays
    frozen: its rows for the five label tokens are a fixed classifier the trained
    layers move `hid` against. Returns the unfrozen params — hand them to
    core.build_optim(slow_params=...) so they train at lr/10 (at the fresh expert's
    rate they blow up within ~200 steps)."""
    lm = policy.model.paligemma_with_expert.paligemma.model.language_model
    ps = []
    for mod in [*lm.layers[-n_layers:], lm.norm]:
        for p in mod.parameters():
            p.requires_grad = True
            ps.append(p)
    policy.model.subtask_ce_attached = True
    return ps
