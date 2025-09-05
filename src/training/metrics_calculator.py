import nltk
import numpy as np
import sacrebleu
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from transformers import PreTrainedTokenizerBase, EvalPrediction

from src.config import ConfigManager
from src.type_defs import MetricScoresType, LoggerType
from src.utils import MemoryManager

# Download necessary NLTK resources (run once or add to the script beginning)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)  # For Russian language support


class MetricsCalculator:
    def __init__(
        self,
        config: ConfigManager,
        tokenizer: PreTrainedTokenizerBase,
        logger: LoggerType,
        use_bertscore: bool = False,
    ) -> None:
        """
        Initializes MetricsCalculator with the required components.

        Args:
            config (ConfigManager): An instance of ConfigManager for configuration.
            tokenizer (PreTrainedTokenizerBase): An instance of PreTrainedTokenizerBase for tokenization.
            logger (LoggerType): A logger for logging operations.
            use_bertscore (bool): Whether to compute BERTScore. Default is False.
        """
        self.config = config
        self.logger = logger
        self.tokenizer = tokenizer
        self.memory_manager = MemoryManager(self.logger)
        self.use_bertscore = use_bertscore

    def compute_metrics(
        self,
        eval_preds: EvalPrediction,
    ) -> MetricScoresType:
        """
        Compute various translation metrics from predictions and labels.

        Args:
            eval_preds (EvalPrediction): A tuple of two numpy arrays containing the predictions and labels.

        Returns:
            MetricScoresType: A dictionary containing the following metrics: BLEU, ChrF, METEOR, and BERTScore F1.
        """
        self.memory_manager.clear()

        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)  # type: ignore

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds_clean = [pred.strip(".").lower() for pred in decoded_preds]
        decoded_labels_clean = [label.strip(".").lower() for label in decoded_labels]

        for pred, ref in zip(decoded_preds_clean[:10], decoded_labels_clean[:10]):
            self.logger.debug(f"\nPRED: {pred}\nREF : {ref}\n{'-'*40}")

        decoded_preds_words = [pred.split() for pred in decoded_preds_clean]
        decoded_labels_words = [label.split() for label in decoded_labels_clean]

        # BLEU
        bleu_result = sacrebleu.corpus_bleu(
            decoded_preds_clean, [decoded_labels_clean]
        ).score

        # ChrF
        chrf_result = sacrebleu.corpus_chrf(
            decoded_preds_clean, [decoded_labels_clean]
        ).score

        # METEOR
        meteor_scores = [
            meteor_score([ref], pred)
            for pred, ref in zip(decoded_preds_words, decoded_labels_words)
        ]
        meteor_avg = float(np.mean(meteor_scores) * 100)

        metrics = {
            "bleu": bleu_result,
            "chrf": chrf_result,
            "meteor": meteor_avg,
        }

        if self.use_bertscore:
            self.memory_manager.clear()

            # BERTScore
            P, R, F1 = bert_score(  # type: ignore
                decoded_preds_clean,
                decoded_labels_clean,
                lang=self.config.lang_config.tgt_lang,
            )
            metrics["bertscore_f1"] = F1.mean().item() * 100

        return metrics
