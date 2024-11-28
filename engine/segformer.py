from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn

from transformers import SegformerModel, SegformerDecodeHead, SegformerConfig

class MultiPathSegformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = MultiPathSegformerEncoder(config)
        self.decoder = MultiPathSegformerDecoder(config)
        self.config = config
    def forward(
        self, 
        inputs: List[torch.FloatTensor],
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
    ):
        pixel_values0 = inputs[0]
        pixel_values1 = inputs[1]
        pixel_values2 = inputs[2]
        encoder_hidden_states = self.encoder.forward(
            pixel_values0=pixel_values0,
            pixel_values1=pixel_values1,
            pixel_values2=pixel_values2,
            output_attentions=self.config.output_attentions
        )

        logits = self.decoder(encoder_hidden_states)
        
        loss = None
        if labels is not None:
            # upsample logits to the images' original size
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            if self.config.num_labels > 1:
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                loss = loss_fct(upsampled_logits, labels)
            elif self.config.num_labels == 1:
                valid_mask = ((labels >= 0) & (labels != self.config.semantic_loss_ignore_index)).float()
                loss_fct = nn.BCEWithLogitsLoss(reduction="none")
                loss = loss_fct(upsampled_logits.squeeze(1), labels.float())
                loss = (loss * valid_mask).mean()
        else:
            logits = nn.functional.interpolate(
                logits, size=inputs[0].shape[-2:], mode="bilinear", align_corners=False
            )
        predicted = upsampled_logits.argmax(dim=1) if labels is not None else logits.argmax(dim=1)
        
        return predicted, loss

class MultiPathSegformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder0 = SegformerModel(config)
        self.encoder1 = SegformerModel(config)
        self.encoder2 = SegformerModel(config)
    def forward(
        self, 
        pixel_values0: torch.FloatTensor, 
        pixel_values1: torch.FloatTensor, 
        pixel_values2: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
    ):
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )

        encoder_hidden_states0 = self.encoder0(
            pixel_values0,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=True,
        ).hidden_states
        encoder_hidden_states1 = self.encoder1(
            pixel_values1,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states
        encoder_hidden_states2 = self.encoder2(
            pixel_values2,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states

        concatenated_hidden_states = list()
        for h0, h1, h2 in zip(encoder_hidden_states0, encoder_hidden_states1, encoder_hidden_states2):
            concatenated_hidden_state = torch.cat((h0, h1, h2), dim=1)
            concatenated_hidden_states.append(concatenated_hidden_state)
        # concatenated_hidden_states = torch.stack(concatenated_hidden_states, dim=0)

        return concatenated_hidden_states

class MultiPathSegformerDecoder(SegformerDecodeHead):
    def __init__(self, config):
        super().__init__(config)
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i] * 3)
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

        self.config = config

class DualPathSegformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = DualPathSegformerEncoder(config)
        self.decoder = DualPathSegformerDecoder(config)
        self.config = config
    def forward(
        self, 
        inputs: List[torch.FloatTensor],
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
    ):
        pixel_values0 = inputs[0]
        pixel_values1 = inputs[1]
        encoder_hidden_states = self.encoder.forward(
            pixel_values0=pixel_values0,
            pixel_values1=pixel_values1,
            output_attentions=self.config.output_attentions
        )

        logits = self.decoder(encoder_hidden_states)
        
        loss = None
        if labels is not None:
            # upsample logits to the images' original size
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            if self.config.num_labels > 1:
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                loss = loss_fct(upsampled_logits, labels)
            elif self.config.num_labels == 1:
                valid_mask = ((labels >= 0) & (labels != self.config.semantic_loss_ignore_index)).float()
                loss_fct = nn.BCEWithLogitsLoss(reduction="none")
                loss = loss_fct(upsampled_logits.squeeze(1), labels.float())
                loss = (loss * valid_mask).mean()
        else:
            logits = nn.functional.interpolate(
                logits, size=inputs[0].shape[-2:], mode="bilinear", align_corners=False
            )
        predicted = upsampled_logits.argmax(dim=1) if labels is not None else logits.argmax(dim=1)
        
        return predicted, loss

class DualPathSegformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder0 = SegformerModel(config)
        self.encoder1 = SegformerModel(config)
    def forward(
        self, 
        pixel_values0: torch.FloatTensor, 
        pixel_values1: torch.FloatTensor, 
        output_attentions: Optional[bool] = None,
    ):
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )

        encoder_hidden_states0 = self.encoder0(
            pixel_values0,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=True,
        ).hidden_states
        encoder_hidden_states1 = self.encoder1(
            pixel_values1,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states

        concatenated_hidden_states = list()
        for h0, h1 in zip(encoder_hidden_states0, encoder_hidden_states1):
            concatenated_hidden_state = torch.cat((h0, h1), dim=1)
            concatenated_hidden_states.append(concatenated_hidden_state)

        return concatenated_hidden_states

class DualPathSegformerDecoder(SegformerDecodeHead):
    def __init__(self, config):
        super().__init__(config)
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i] * 2)
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

        self.config = config

class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """
    def __init__(self, config: SegformerConfig, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states