from masa.algorithms.tabular.recreg import RECERG

class RECREG_MODEL_BASED(RECERG):

    def __init__(
        self,
        *args,
        mode: str = "model_based",
        model_checking: str = "exact",
        **kwargs,
    ):

        super().__init__(
            *args,
            mode=mode,
            model_checking=model_checking,
            **kwargs,
        )