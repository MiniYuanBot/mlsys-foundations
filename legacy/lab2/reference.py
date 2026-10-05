"""Readable baseline A; torch.softmax is baseline B."""
def explicit_softmax(x):
    shifted = x - x.max(dim=-1, keepdim=True).values
    numerator = shifted.exp()
    return numerator / numerator.sum(dim=-1, keepdim=True)

