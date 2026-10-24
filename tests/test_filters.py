
import pandas as pd
from src.filters.metadata_filters import apply_filters


def test_category_and_zone_filters():
    products = pd.DataFrame({
        "category": ["Luggage", "Luggage", "Books"],
        "zone": ["ZONE-01", "ZONE-02", "ZONE-01"],
        "price": [100.0, 200.0, 50.0],
    })

    mask = apply_filters(
        products,
        {"category": "Luggage", "zone": "ZONE-01"},
    )

    assert mask.tolist() == [True, False, False]


def test_price_range():
    products = pd.DataFrame({
        "category": ["A", "A", "B"],
        "zone": ["ZONE-01", "ZONE-01", "ZONE-02"],
        "price": [50.0, 100.0, 200.0],
    })

    mask = apply_filters(
        products,
        {"min_price": 75, "max_price": 150},
    )

    assert mask.tolist() == [False, True, False]


def test_no_filters_matches_every_product():
    products = pd.DataFrame({"price": [10, 20, 30]})

    assert apply_filters(products).tolist() == [True, True, True]
