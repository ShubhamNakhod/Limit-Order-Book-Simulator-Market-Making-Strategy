# tests/test_lob.py

import math
import pytest

from lob.matching_engine import OrderBook, Order, Side
from strategy.rule_based_mm import RuleBasedMarketMaker


def make_seeded_book() -> OrderBook:
    """
    Create an OrderBook with passive liquidity on both sides
    so the mid-price is defined.
    """
    ob = OrderBook()
    ob.insert_order(Order(order_id="seed_bid", side=Side.BUY,  price=99.0, quantity=100, timestamp=0))
    ob.insert_order(Order(order_id="seed_ask", side=Side.SELL, price=101.0, quantity=100, timestamp=0))
    return ob


def test_exact_price_match():
    ob = OrderBook()
    buy  = Order(order_id="b1", side=Side.BUY,  price=100.0, quantity=10, timestamp=1.0)
    sell = Order(order_id="s1", side=Side.SELL, price=100.0, quantity=10, timestamp=2.0)

    trades_buy = ob.insert_order(buy)
    assert trades_buy == []  # no liquidity yet on the other side

    trades_sell = ob.insert_order(sell)
    assert isinstance(trades_sell, list)
    assert len(trades_sell) == 1

    tr = trades_sell[0]
    assert tr.price == 100.0
    assert tr.quantity == 10
    assert tr.buy_order_id == "b1"
    assert tr.sell_order_id == "s1"

    # both orders fully filled -> gone from the map
    assert "b1" not in ob.order_map
    assert "s1" not in ob.order_map


def test_partial_fill_and_remaining():
    ob = OrderBook()
    # Large buy, then a small crossing sell
    buy  = Order(order_id="b1", side=Side.BUY,  price=100.0, quantity=20, timestamp=1.0)
    sell = Order(order_id="s1", side=Side.SELL, price= 99.5, quantity= 5, timestamp=2.0)

    ob.insert_order(buy)
    trades = ob.insert_order(sell)

    assert isinstance(trades, list)
    assert len(trades) == 1
    tr = trades[0]
    assert tr.quantity == 5
    assert tr.price == 99.5

    # remaining quantity on the original buy should be 15
    assert "b1" in ob.order_map
    remaining_buy = ob.order_map["b1"]
    assert remaining_buy.quantity == 15


def test_cancel_order_prevents_matching():
    ob = OrderBook()
    buy = Order(order_id="b1", side=Side.BUY, price=100.0, quantity=10, timestamp=1.0)
    ob.insert_order(buy)
    assert "b1" in ob.order_map

    # cancel the resting bid
    cancelled = ob.cancel_order("b1")
    assert cancelled is True
    assert "b1" not in ob.order_map

    # a crossing sell should NOT match now
    sell = Order(order_id="s1", side=Side.SELL, price=99.5, quantity=10, timestamp=2.0)
    trades = ob.insert_order(sell)
    assert trades == []


def test_rule_based_mm_basic_behavior():
    ob = make_seeded_book()
    mm = RuleBasedMarketMaker(
        order_book=ob,
        half_spread=1.0,
        alpha=0.05,
        order_size=5,
        max_inventory=50,
    )

    # Step once: should place quotes and record history
    mm.step(timestamp=1.0)
    history = mm.get_history()
    assert isinstance(history, list)
    assert len(history) == 1
    state = history[0]

    # Inventory bounded and PnL consistent
    assert abs(state.inventory) <= mm.max_inventory
    expected_pnl = mm.cash + mm.inventory * state.mid_price
    assert math.isclose(state.pnl, expected_pnl, rel_tol=1e-8)

    # Advance a few more steps; history grows and bounds hold
    for t in [2.0, 3.0, 4.0]:
        mm.step(timestamp=t)
    assert len(mm.get_history()) == 4
    final = mm.get_history()[-1]
    assert abs(final.inventory) <= mm.max_inventory
