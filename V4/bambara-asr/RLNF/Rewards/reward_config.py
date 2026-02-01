from transformers import PretrainedConfig, Wav2Vec2Config, WavLMConfig

class RewardConfig(PretrainedConfig):
    """
    Configuration class for the RewardModel.

    This configuration defines how the RewardModel initializes its audio feature extractor 
    and architecture. It extends `PretrainedConfig`, meaning it can be saved, loaded, and shared 
    just like any Hugging Face model config.

    Args:
    
            
        n_mel (`int`, *optional*, defaults to `80`):
            Number of Mel filter banks in the spectrogram input (used only if `use_pre="default"`).

        vocab_size (`int`, *optional*, defaults to `5000`):
            Size of the vocabulary for the output tokens (used only if `use_pre="default"`).

        embed_dim (`int`, *optional*, defaults to `128`):
            Dimensionality of the token or feature embeddings.

        lstm_hidden (`int`, *optional*, defaults to `128`):
            Hidden size of each LSTM layer in the sequence encoder.

        lstm_layers (`int`, *optional*, defaults to `1`):
            Number of LSTM layers stacked in the encoder.

        audio_conv_channels (`int`, *optional*, defaults to `128`):
            Number of output channels for the convolutional layers processing audio features.

        audio_conv_layers (`int`, *optional*, defaults to `3`):
            Number of convolutional layers applied to the input spectrogram.

        head_hidden (`int`, *optional*, defaults to `256`):
            Hidden size of the final fully-connected (reward head) layer.

        dropout (`float`, *optional*, defaults to `0.3`):
            Dropout probability applied in linear and LSTM layers.

        override (`dict`, *optional*):
            Dictionary of parameter overrides. Any field in the base config 
            will be replaced by the provided value.
        
        tokenizer_path(`str` *optional*):
            Path to the tokenizer to use(here probably a sentenpiece tokenizer).

        **kwargs:
            Additional arguments passed to `PretrainedConfig`.

    Attributes:
        pretrained_config (`dict`): The resolved configuration dictionary.
        override (`dict`): User-provided overrides applied to the base configuration.
    ---
    
    ## Examples

    >>> # Default custom configuration
    >>> config = RewardConfig()
    >>> config.pretrained_config
    {'n_mel': 80, 'vocab_size': 5000, 'embed_dim': 128,
     'lstm_hidden': 128, 'lstm_layers': 1, 'audio_conv_channels': 128,
     'audio_conv_layers': 3, 'head_hidden': 256, 'dropout': 0.3}
        
    """

    model_type = "reward-model"

    def __init__(
        self,
        override: dict = None,
        tokenizer_path : str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.override = override or {}    
            
        self.tokenizer_path = tokenizer_path

       
        base_cfg = {
            "n_mel": 64,
            "vocab_size": 2048,
            "sample_rate" : 16_000,
            "n_fft" : 1024,
            "hop_length" : 256,
            "embed_dim": 128,
            "lstm_hidden": 128,
            "lstm_layers": 1,
            "audio_conv_channels": 128,
            "audio_conv_layers": 3,
            "kernel_size" : 5,
            "stride" : 1 ,
            "padding" : 2, 
            "head_hidden": 256,
            "dropout": 0.1,
            "pad_token_id" : 1,
            "chunk_length" : 30,
            "padding_value" : 0,
            "dither" : 0.0, 
            "return_attention_mask" : True
              
        }
            
        base_cfg.update(self.override)
        self.pretrained_config = base_cfg
