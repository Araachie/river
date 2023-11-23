class RunningMean:
    def __init__(self):
        super(RunningMean, self).__init__()

        self.values = {}

    def update(self, values: dict):
        for k, v in values.items():
            if k not in self.values:
                self.values[k] = (v, 1)
            else:
                m, n = self.values[k]
                self.values[k] = ((n * m + v) / (n + 1), n + 1)

    def get_values(self):
        return {k: v[0] for k, v in self.values.items()}
