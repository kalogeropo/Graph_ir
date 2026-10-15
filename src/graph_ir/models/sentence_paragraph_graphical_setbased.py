from math import isfinite
import warnings

from graph_ir.graphs.operations import fixed_windows, validate_window
from graph_ir.models.graphical_setbased import GSBModel


def _validate_blend(sentence_weight, paragraph_weight):
    """Reject blends that cannot produce a usable co-occurrence graph.

    A negative coefficient makes negative edge weights, which the union graph's
    node weighting is not defined for, and two zeros make every edge zero: the
    graph comes out empty, _nwk comes out empty, and _model_func then returns
    all-ones, silently turning this model into SetBasedModel.
    """
    for name, weight in (
        ("sentence_weight", sentence_weight), ("paragraph_weight", paragraph_weight),
    ):
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"{name} must be a real number")
        if not isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if sentence_weight == 0 and paragraph_weight == 0:
        raise ValueError(
            "sentence_weight and paragraph_weight cannot both be zero: "
            "the resulting graph has no edges"
        )


class SentenceParagraphGSBModel(GSBModel):
    """GSB variant blending two fixed co-occurrence windows of different sizes.

    Builds two co-occurrence matrices per document from the same token stream --
    a small "sentence-scale" window (sen_window) and a larger "paragraph-scale"
    window (par_window) -- and combines them as
    ``sentence_weight * sentence_matrix + paragraph_weight * paragraph_matrix``.

    Both scales are plain fixed/fractional windows over document.terms. Neither
    looks at sentence punctuation, and that is deliberate: real sentence
    boundaries can only be recovered from the raw text, which means they depend
    on punctuation surviving every earlier preprocessing step. Tokenizers
    routinely strip it (preprocess_text does), and documents loaded from a
    pre-tokenized field never had it, so a punctuation-dependent window
    degrades silently to one window spanning the whole document. Making a
    two-scale model rest on that is not worth it -- SentenceGSBModel is the one
    place that genuinely uses sentence boundaries, and it is explicit about it.

    This is the old project's SenParConWind shape: two fixed window sizes
    (its sen_window / par_window) blended by coefficients. The old code's
    `dot_split` flag, which optionally swapped the small scale for real
    sentence boundaries, is not reproduced -- SentenceGSBModel covers that
    case on its own.

    Args:
        sen_window: Small-scale window. Positive int token count, or a finite
            positive float fraction of the document length.
        par_window: Large-scale window, same units as sen_window.
        sentence_weight: Coefficient applied to the sen_window matrix.
        paragraph_weight: Coefficient applied to the par_window matrix.
    """

    def __init__(self,
                 collection,
                 sen_window: int | float = 8,
                 par_window: int | float = 32,
                 sentence_weight: float = 1.0,
                 paragraph_weight: float = 0.05,
                 h_val: int = 1,
                 k_core_bool: bool = False):
        validate_window(sen_window, "sen_window")
        validate_window(par_window, "par_window")
        _validate_blend(sentence_weight, paragraph_weight)
        if sen_window == par_window:
            warnings.warn(
                f"sen_window and par_window are both {sen_window}, so both scales "
                f"produce the same matrix and the blend collapses to a single "
                f"window scaled by sentence_weight + paragraph_weight",
                stacklevel=2,
            )
        self.sen_window = sen_window
        self.par_window = par_window
        self.sentence_weight = sentence_weight
        self.paragraph_weight = paragraph_weight
        super().__init__(collection, k_core_bool=k_core_bool, h_val=h_val)

    def _window_sets(self, document):
        """The two fixed window scales, with their blend coefficients."""
        return [
            (self.sentence_weight, fixed_windows(document.terms, self.sen_window)),
            (self.paragraph_weight, fixed_windows(document.terms, self.par_window)),
        ]
