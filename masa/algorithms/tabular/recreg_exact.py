from masa.algorithms.tabular.recreg import RECERG

class RECREG_EXACT(RECERG):

    def __init__(
        self,
        *args,
        mode: str = "exact",
        model_checking: str = "exact",
        **kwargs,
    ):

        super().__init__(
            *args,
            mode=mode,
            model_checking=model_checking,
            **kwargs,
        )