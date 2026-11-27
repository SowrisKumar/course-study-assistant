"""Keep terminal and server output clean: no model-loading progress bars or library notices.

Import this before the heavy libraries (sentence-transformers, torch). Used by the chat CLI and the API.
"""

import logging
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
for _noisy in ("torch.distributed.elastic.multiprocessing.redirects", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)
