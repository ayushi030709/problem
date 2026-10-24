
import pandas as pd


def apply_filters(products, filters=None):
    """Return a Boolean mask for all supplied AND conditions."""
    mask = pd.Series(True, index=products.index)

    if not filters:
        return mask

    if "category" in filters:
        mask &= products["category"].eq(filters["category"])

    if "zone" in filters:
        mask &= products["zone"].eq(filters["zone"])

    if "min_price" in filters:
        mask &= products["price"].ge(filters["min_price"])

    if "max_price" in filters:
        mask &= products["price"].le(filters["max_price"])

    return mask
