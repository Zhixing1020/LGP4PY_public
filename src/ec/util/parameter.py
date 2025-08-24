

class Parameter:
    def __init__(self, base: str):
        self.path = base

    def push(self, sub: str):
        return Parameter(f"{self.path}.{sub}")

    def __str__(self):
        return self.path

