import contextvars
import logging


dataset_var = contextvars.ContextVar("dataset", default="-")
scenario_var = contextvars.ContextVar("scenario", default="-")

_FORMAT = "%(levelname)s bin_factory%(context)s: %(message)s"


def _inject(record):
    dataset = dataset_var.get()
    scenario = scenario_var.get()
    record.context = f" dataset={dataset} scenario={scenario}" if (dataset != "-" or scenario != "-") else ""
    return True


log = logging.getLogger("bin_factory")
log.setLevel("INFO")
log.propagate = False
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter(_FORMAT))
_h.addFilter(_inject)
log.addHandler(_h)


def bind(dataset="-", scenario="-"):
    return dataset_var.set(dataset), scenario_var.set(scenario)


def unbind(tokens):
    dataset_var.reset(tokens[0])
    scenario_var.reset(tokens[1])
