import contextvars
import logging


dataset_var = contextvars.ContextVar("dataset", default="-")
scenario_var = contextvars.ContextVar("scenario", default="-")

_FORMAT = "%(levelname)s %(name)s dataset=%(dataset)s scenario=%(scenario)s: %(message)s"


class _ContextFilter(logging.Filter):
    def filter(self, record):
        record.dataset = dataset_var.get()
        record.scenario = scenario_var.get()
        return True


def setup_logging(level="INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level)
    if any(getattr(h, "_bin_factory_ctx", False) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.addFilter(_ContextFilter())
    handler._bin_factory_ctx = True
    root.addHandler(handler)


def bind(dataset: str = "-", scenario: str = "-"):
    return dataset_var.set(dataset), scenario_var.set(scenario)


def unbind(tokens) -> None:
    dataset_token, scenario_token = tokens
    dataset_var.reset(dataset_token)
    scenario_var.reset(scenario_token)
