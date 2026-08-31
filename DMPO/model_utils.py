def align_logits_and_targets(model_type, logits, input_ids, masked_index):
    """Align denoising logits, targets, and masks for a model family."""
    if model_type == "dream":
        return logits[:, :-1].contiguous(), input_ids[:, 1:].contiguous(), masked_index[:, 1:].contiguous()
    return logits, input_ids, masked_index
