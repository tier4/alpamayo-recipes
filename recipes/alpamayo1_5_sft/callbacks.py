# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer callbacks for Alpamayo-1.5 SFT recipes."""

from __future__ import annotations

from typing import Any

from transformers import TrainerCallback
from transformers import TrainerControl
from transformers import TrainerState
from transformers import TrainingArguments

from alpamayo_r1.common import logging

logger = logging.RankedLogger(__name__, rank_zero_only=True)


class DriveLMEpochSamplerCallback(TrainerCallback):
    """Refresh DriveLM curated training samples at the start of each epoch."""

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        train_dataloader = kwargs.get("train_dataloader")
        dataset = getattr(train_dataloader, "dataset", None)
        set_epoch = getattr(dataset, "set_epoch", None)
        if callable(set_epoch):
            epoch = int(state.epoch or 0)
            set_epoch(epoch)
            logger.info("[DriveLMEpochSamplerCallback] refreshed train dataset for epoch %d", epoch)
        return control
