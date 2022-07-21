class ShieldingError(Exception):
    pass


class UnsafeStateError(ShieldingError):
    pass


class InvalidOutputError(ShieldingError):
    pass


class InvalidInputError(ShieldingError):
    pass


class UnknownOutputError(ShieldingError):
    pass


class UnknownStateError(ShieldingError):
    pass


class InvalidMoveError(ShieldingError):
    pass


class InvalidMoveWithShieldError(ShieldingError):
    pass
