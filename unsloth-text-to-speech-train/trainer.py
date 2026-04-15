from unsloth import FastModel, FastLanguageModel

from abc import ABC, abstractmethod
from transformers import CsmForConditionalGeneration
import torch
from transformers import AutoProcessor
from snac import SNAC


def _patch_csm_depth_decoder_inplace(model):
    """
    Patch the CSM depth decoder to fix the in-place operation error on inputs_embeds.
    
    In Transformers 5.x, CsmDepthDecoderModel.forward() does:
        inputs_embeds[:, 0] = backbone_last_hidden_state
    which fails during training because inputs_embeds is a leaf variable requiring grad.
    
    This patch wraps the depth_decoder.model.forward to clone inputs_embeds first.
    Must be called AFTER model loading (after FastModel.from_pretrained and get_peft_model)
    since Unsloth may overwrite earlier patches.
    """
    try:
        # Navigate to the actual depth_decoder model
        # The model may be wrapped in PEFT, so we need to find the base model
        base_model = model
        while hasattr(base_model, 'model') and not isinstance(base_model, CsmForConditionalGeneration):
            base_model = base_model.model
        
        # base_model should now be CsmForConditionalGeneration
        if not hasattr(base_model, 'depth_decoder'):
            print("⚠️  Could not find depth_decoder on model, skipping in-place fix")
            return
        
        depth_decoder_model = base_model.depth_decoder.model  # CsmDepthDecoderModel
        _original_forward = depth_decoder_model.forward.__func__ if hasattr(depth_decoder_model.forward, '__func__') else depth_decoder_model.forward
        
        import functools
        import types
        
        @functools.wraps(_original_forward)
        def _patched_forward(self, input_ids=None, backbone_last_hidden_state=None,
                             attention_mask=None, position_ids=None,
                             past_key_values=None, inputs_embeds=None,
                             use_cache=None, **kwargs):
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds.clone()
            elif input_ids is not None:
                # Replicate the embedding logic but with a clone before in-place ops
                from transformers.cache_utils import DynamicCache
                from transformers.masking_utils import create_causal_mask
                from transformers.modeling_outputs import BaseModelOutputWithPast
                
                if use_cache and past_key_values is None:
                    past_key_values = DynamicCache(config=self.config)
                
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                inputs_seq_length = input_ids.shape[1]
                device = input_ids.device
                pos_ids = torch.arange(past_seen_tokens, past_seen_tokens + inputs_seq_length, device=device)
                
                codebook_idxs = torch.clamp(pos_ids - 1, min=0)
                offset = codebook_idxs * self.vocab_size
                inputs_embeds = self.embed_tokens(input_ids + offset).clone()
                
                if backbone_last_hidden_state is not None and pos_ids[0] == 0:
                    inputs_embeds[:, 0] = backbone_last_hidden_state
                
                inputs_embeds = self.inputs_embeds_projector(inputs_embeds)
                
                causal_mask = create_causal_mask(
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=pos_ids,
                )
                
                hidden_states = inputs_embeds
                pos_ids = pos_ids.unsqueeze(0)
                position_embeddings = self.rotary_emb(hidden_states, position_ids=pos_ids)
                
                for decoder_layer in self.layers[:self.config.num_hidden_layers]:
                    hidden_states = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=pos_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )
                
                hidden_states = self.norm(hidden_states)
                return BaseModelOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values=past_key_values if use_cache else None,
                )
            
            # Fallback: neither input_ids nor inputs_embeds
            return _original_forward(
                self, input_ids=input_ids,
                backbone_last_hidden_state=backbone_last_hidden_state,
                attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                use_cache=use_cache, **kwargs
            )
        
        # Bind the patched method to the instance
        depth_decoder_model.forward = types.MethodType(_patched_forward, depth_decoder_model)
        print("✅ Applied post-load CSM patch: depth_decoder.model.forward (in-place fix)")
        
    except Exception as e:
        print(f"⚠️  Could not apply post-load CSM depth_decoder patch: {e}")
        import traceback
        traceback.print_exc()


class AudioTrainerBase(ABC):
    def __init__(
        self,
        model_name,
        context_length,
        device,
        speaker_key,
        lora_r,
        lora_alpha,
        lora_dropout,
        sampling_rate,
        max_audio_length,
        audio_column_name="audio",
        text_column_name="text",
    ):
        self.model_name = model_name
        self.context_length = context_length
        self.device = device
        self.speaker_key = speaker_key
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.sampling_rate = sampling_rate
        self.max_audio_length = max_audio_length
        self.audio_column_name = audio_column_name
        self.text_column_name = text_column_name
        self.lora_target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    @abstractmethod
    def preprocess_dataset(self, example):
        pass


class CsmAudioTrainer(AudioTrainerBase):
    def __init__(
        self,
        model_name,
        context_length,
        device,
        speaker_key,
        lora_r,
        lora_alpha,
        lora_dropout,
        sampling_rate,
        max_audio_length,
        audio_column_name="audio",
        text_column_name="text",
    ):
        super().__init__(
            model_name,
            context_length,
            device,
            speaker_key,
            lora_r,
            lora_alpha,
            lora_dropout,
            sampling_rate,
            max_audio_length,
            audio_column_name,
            text_column_name,
        )
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.context_length,
            dtype=None,  # Leave as None for auto-detection
            auto_model=CsmForConditionalGeneration,
            load_in_4bit=False,  # Keep this set to False because voice models are small, so we can maintain high quality results.
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = FastModel.get_peft_model(
            self.model,
            r=lora_r,
            target_modules=self.lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",  # Supports any, but = "none" is optimized
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )
        
        # Apply post-load patches for Transformers 5.x compatibility
        # Must be done AFTER FastModel.get_peft_model since Unsloth may overwrite forward methods
        _patch_csm_depth_decoder_inplace(self.model)
        
        num_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Trainable parameters: {num_trainable}")

    def preprocess_dataset(self, example):
        conversation = [
            {
                "role": str(example[self.speaker_key]),
                "content": [
                    {"type": "text", "text": example[self.text_column_name]},
                    {"type": "audio", "path": example[self.audio_column_name]["array"]},
                ],
            }
        ]

        try:
            model_inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
                output_labels=True,
                text_kwargs={
                    "padding": "max_length",  # pad to the max_length
                    "max_length": self.context_length,  # this should be the max length of audio
                    "pad_to_multiple_of": 8,  # Pad so length is a multiple of 8 (for efficiency)
                    "padding_side": "right",
                },
                audio_kwargs={
                    "sampling_rate": self.sampling_rate,
                    "max_length": self.max_audio_length,  # max input_values length of the whole dataset
                    "padding": "max_length",
                },
                common_kwargs={"return_tensors": "pt"},
            )
        except Exception as e:
            print(
                f"Error processing example with text '{example[self.text_column_name][:50]}...': {e}"
            )
            return None

        required_keys = [
            "input_ids",
            "attention_mask",
            "labels",
            "input_values",
            "input_values_cutoffs",
        ]
        processed_example = {}
        for key in required_keys:
            if key not in model_inputs:
                print(
                    f"Warning: Required key '{key}' not found in processor output for example."
                )
                return None

            value = model_inputs[key][0]
            # Detach and clone tensors to avoid in-place operation issues
            # Convert to contiguous CPU tensors to prevent view-related issues
            if isinstance(value, torch.Tensor):
                value = value.detach().clone().cpu()
            processed_example[key] = value

        if not all(
            isinstance(processed_example[key], torch.Tensor)
            for key in processed_example
        ):
            print(
                f"Error: Not all required keys are tensors in final processed example. Keys: {list(processed_example.keys())}"
            )
            return None

        return processed_example


class OrpheusAudioTrainer(AudioTrainerBase):
    def __init__(
        self,
        model_name,
        context_length,
        device,
        speaker_key,
        lora_r,
        lora_alpha,
        lora_dropout,
        sampling_rate,
        max_audio_length,
        batch_size,
        audio_column_name="audio",
        text_column_name="text",
    ):
        super().__init__(
            model_name,
            context_length,
            device,
            speaker_key,
            lora_r,
            lora_alpha,
            lora_dropout,
            sampling_rate,
            max_audio_length,
            audio_column_name,
            text_column_name,
        )
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(
            self.device
        )
        self.model, self.processor = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.context_length,
            dtype=None,
            load_in_4bit=False,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_r,
            target_modules=self.lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",  # Supports any, but = "none" is optimized
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )
        num_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Trainable parameters: {num_trainable}")

        # Define special tokens
        self.tokenizer_length = 128256
        self.start_of_text = 128000
        self.end_of_text = 128009
        self.start_of_speech = self.tokenizer_length + 1
        self.end_of_speech = self.tokenizer_length + 2
        self.start_of_human = self.tokenizer_length + 3
        self.end_of_human = self.tokenizer_length + 4
        self.start_of_ai = self.tokenizer_length + 5
        self.end_of_ai = self.tokenizer_length + 6
        self.audio_tokens_start = self.tokenizer_length + 10
        self.pad_token = self.tokenizer_length + 7
        self.ds_sample_rate = 24000
        self.batch_size = batch_size

    def _tokenize_audio(self, waveform):
        """Convert audio waveform to SNAC tokens."""
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32)

        waveform = waveform.unsqueeze(0).to(self.device)

        # Generate SNAC codes
        with torch.inference_mode():
            codes = self.snac_model.encode(waveform)

        # Interleave codes according to Orpheus format
        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.extend(
                [
                    codes[0][0][i].item() + 128266,
                    codes[1][0][2 * i].item() + 128266 + 4096,
                    codes[2][0][4 * i].item() + 128266 + (2 * 4096),
                    codes[2][0][(4 * i) + 1].item() + 128266 + (3 * 4096),
                    codes[1][0][(2 * i) + 1].item() + 128266 + (4 * 4096),
                    codes[2][0][(4 * i) + 2].item() + 128266 + (5 * 4096),
                    codes[2][0][(4 * i) + 3].item() + 128266 + (6 * 4096),
                ]
            )

        return all_codes

    def _remove_duplicate_frames(self, codes_list):
        """Remove consecutive duplicate audio frames to reduce redundancy."""
        if len(codes_list) % 7 != 0:
            raise ValueError("Input list length must be divisible by 7")

        result = codes_list[:7]

        for i in range(7, len(codes_list), 7):
            current_first = codes_list[i]
            previous_first = result[-7]

            if current_first != previous_first:
                result.extend(codes_list[i : i + 7])

        return result

    def preprocess_dataset(self, example):
        """
        Preprocess a single example for Orpheus training.
        """
        try:
            # Extract and tokenize audio
            audio_array = example[self.audio_column_name]["array"]
            codes_list = self._tokenize_audio(audio_array)

            if not codes_list:
                print(
                    f"Warning: Empty codes list for example with text '{example[self.text_column_name][:50]}...'"
                )
                return None

            # Remove duplicate frames for efficiency
            codes_list = self._remove_duplicate_frames(codes_list)

            # Create text prompt (multi-speaker or single-speaker)
            if self.speaker_key in example and example[self.speaker_key]:
                text_prompt = (
                    f"{example[self.speaker_key]}: {example[self.text_column_name]}"
                )
            else:
                text_prompt = example[self.text_column_name]

            text_ids = self.processor.encode(text_prompt, add_special_tokens=True)
            text_ids.append(self.end_of_text)

            # Construct input sequence with special tokens
            input_ids = (
                [self.start_of_human]
                + text_ids
                + [self.end_of_human]
                + [self.start_of_ai]
                + [self.start_of_speech]
                + codes_list
                + [self.end_of_speech]
                + [self.end_of_ai]
            )

            # Use tokenizer to handle truncation and padding properly
            if len(input_ids) > self.context_length:
                input_ids = input_ids[: self.context_length]

            labels = input_ids.copy()

            # Pad to context_length using the pad_token
            padding_length = self.context_length - len(input_ids)
            if padding_length > 0 and self.batch_size > 1:
                input_ids.extend([self.pad_token] * padding_length)
                labels.extend([-100] * padding_length)
                attention_mask = [1] * (len(input_ids) - padding_length) + [
                    0
                ] * padding_length
            else:
                attention_mask = [1] * len(input_ids)

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

        except Exception as e:
            print(
                f"Error processing example with text '{example[self.text_column_name][:50]}...': {e}"
            )
            return None
