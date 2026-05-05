"""Vision Auto-Rubric package."""


def __getattr__(name):
    if name == "Judger":
        from judger import Judger

        return Judger
    raise AttributeError(name)
