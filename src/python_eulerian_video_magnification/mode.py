import enum


@enum.unique
class Mode(enum.Enum):
    COLOR   = 1
    MOTION  = 2
    RIESZ   = 3

    def __str__(self):
        return str(self.name)

    @staticmethod
    def from_string(s):
        try:
            return Mode[s.upper()]
        except KeyError as ex_error:
            raise ValueError(ex_error)
