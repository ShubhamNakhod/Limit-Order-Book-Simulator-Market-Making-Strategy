# strategy/rule_based_mm.py

import uuid
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from lob.matching_engine import OrderBook, Order, Side, Trade


@dataclass
class MMState:
    timestamp: float
    inventory: int = 0
    cash: float = 0.0
    mid_price: float = 0.0
    pnl: float = 0.0  # mark-to-market


class RuleBasedMarketMaker:
    """
    Inventory-skewed market maker:
      - Mid from top-of-book
      - Quotes (mid ± half_spread) + skew, skew = alpha * inventory
      - Quotes rest for a few steps (TTL) so they can be hit/lifted
      - Tracks cash, inventory, and P&L
    """

    def __init__(
        self,
        order_book: OrderBook,
        half_spread: float = 0.05,   # tighter -> more interaction
        alpha: float = 0.20,         # stronger skew -> faster rebalancing
        order_size: int = 10,
        max_inventory: int = 60,
        cross_prob: float = 0.20,    # small chance to cross and force a fill
        quote_ttl_steps: int = 3,    # quotes live for N steps before refresh
    ):
        self.ob = order_book
        self.half_spread = half_spread
        self.alpha = alpha
        self.order_size = order_size
        self.max_inventory = max_inventory
        self.cross_prob = cross_prob
        self.quote_ttl_steps = quote_ttl_steps

        self.current_bid_id: Optional[str] = None
        self.current_ask_id: Optional[str] = None
        self.bid_ttl = 0
        self.ask_ttl = 0

        self.inventory = 0
        self.cash = 0.0
        self.last_mid: float = 0.0

        self.history: List[MMState] = []

    # ----------------- helpers -----------------

    def _clean_peek_best(self) -> Tuple[Optional[float], Optional[float]]:
        """Return (best_bid, best_ask) after lazily cleaning heaps."""
        import heapq

        best_bid = None
        best_ask = None

        while self.ob.bids:
            neg_price, ts, oid = self.ob.bids[0]
            if oid in self.ob.order_map:
                best_bid = -neg_price
                break
            heapq.heappop(self.ob.bids)

        while self.ob.asks:
            price, ts, oid = self.ob.asks[0]
            if oid in self.ob.order_map:
                best_ask = price
                break
            heapq.heappop(self.ob.asks)

        return best_bid, best_ask

    def _compute_mid(self) -> float:
        """Mid from best bid/ask (fallback to last seen)."""
        best_bid, best_ask = self._clean_peek_best()

        if best_bid is not None and best_ask is not None:
            self.last_mid = 0.5 * (best_bid + best_ask)
        elif best_bid is not None:
            self.last_mid = best_bid + self.half_spread
        elif best_ask is not None:
            self.last_mid = best_ask - self.half_spread
        return self.last_mid

    def _cancel_if_expired(self):
        # decrement TTLs; cancel when they expire
        if self.current_bid_id is not None:
            self.bid_ttl -= 1
            if self.bid_ttl <= 0:
                self.ob.cancel_order(self.current_bid_id)
                self.current_bid_id = None
        if self.current_ask_id is not None:
            self.ask_ttl -= 1
            if self.ask_ttl <= 0:
                self.ob.cancel_order(self.current_ask_id)
                self.current_ask_id = None

    def _process_trades(self, trades: Optional[List[Trade]]) -> None:
        if not trades:
            return
        for tr in trades:
            if tr.buy_order_id == self.current_bid_id:
                self.inventory += tr.quantity
                self.cash -= tr.price * tr.quantity
            elif tr.sell_order_id == self.current_ask_id:
                self.inventory -= tr.quantity
                self.cash += tr.price * tr.quantity

    # ----------------- main step -----------------

    def _place_or_refresh_quotes(self, timestamp: float) -> None:
        mid = self._compute_mid()
        best_bid, best_ask = self._clean_peek_best()

        # Base symmetric quotes with inventory skew
        skew = self.alpha * self.inventory  # positive inv -> push bid up, ask out
        bid_price = mid - self.half_spread + skew
        ask_price = mid + self.half_spread + skew

        # Occasionally cross to force fills (demo)
        if random.random() < self.cross_prob:
            if best_ask is not None and self.inventory < self.max_inventory:
                bid_price = max(bid_price, best_ask)   # marketable bid
            if best_bid is not None and self.inventory > -self.max_inventory:
                ask_price = min(ask_price, best_bid)   # marketable ask

        can_buy = self.inventory < self.max_inventory
        can_sell = self.inventory > -self.max_inventory

        # (Re)post bid if none alive
        if can_buy and self.current_bid_id is None:
            bid_id = str(uuid.uuid4())
            self.current_bid_id = bid_id
            self.bid_ttl = self.quote_ttl_steps
            trades = self.ob.insert_order(Order(
                order_id=bid_id, side=Side.BUY, price=bid_price,
                quantity=self.order_size, timestamp=timestamp
            )) or []
            self._process_trades(trades)

        # (Re)post ask if none alive
        if can_sell and self.current_ask_id is None:
            ask_id = str(uuid.uuid4())
            self.current_ask_id = ask_id
            self.ask_ttl = self.quote_ttl_steps
            trades = self.ob.insert_order(Order(
                order_id=ask_id, side=Side.SELL, price=ask_price,
                quantity=self.order_size, timestamp=timestamp
            )) or []
            self._process_trades(trades)

    def step(self, timestamp: float) -> None:
        """
        One epoch:
          1) Let existing quotes age; cancel when TTL hits 0
          2) Place fresh quotes if needed
          3) Log state
        """
        self._cancel_if_expired()
        self._place_or_refresh_quotes(timestamp)

        mid = self._compute_mid()
        pnl = self.cash + self.inventory * mid

        self.history.append(
            MMState(
                timestamp=timestamp,
                inventory=self.inventory,
                cash=self.cash,
                mid_price=mid,
                pnl=pnl,
            )
        )

    def get_history(self) -> List[MMState]:
        return self.history


# ----------------- demo runner -----------------

if __name__ == "__main__":
    """
    Demo with balanced external pressure around the *current mid*:
      - seeds a small ladder
      - injects symmetric buys/sells near mid each step (random sizes)
      - MM uses TTL quotes + occasional marketable orders
      - writes data/history_rule_based.json for the dashboard
    """
    import os, json

    ob = OrderBook()
    mm = RuleBasedMarketMaker(
        order_book=ob,
        half_spread=0.05,
        alpha=0.20,
        order_size=6,
        max_inventory=40,
        cross_prob=0.20,
        quote_ttl_steps=3,
    )

    # Seed a small depth ladder on both sides (price-time FIFO still applies)
    for i, p in enumerate([99.7, 99.6, 99.5, 99.4], start=1):
        ob.insert_order(Order(order_id=f"seed_bid_{i}", side=Side.BUY,  price=p, quantity=60, timestamp=0 + i*0.001))
    for i, p in enumerate([100.3, 100.4, 100.5, 100.6], start=1):
        ob.insert_order(Order(order_id=f"seed_ask_{i}", side=Side.SELL, price=p, quantity=60, timestamp=0 + i*0.001))

    # Run 120 steps with symmetric, mid-anchored external orders
    for t in range(1, 121):
        mid = mm._compute_mid()
        # External sells that *cross* best bid slightly
        if random.random() < 0.55:
            ob.insert_order(Order(
                order_id=f"ext_sell_{t}",
                side=Side.SELL,
                price=mid - random.uniform(0.05, 0.20),  # near/below mid -> likely marketable vs bids
                quantity=random.choice([4, 6, 8, 10]),
                timestamp=t + 0.10,
            ))
        # External buys that *cross* best ask slightly
        if random.random() < 0.55:
            ob.insert_order(Order(
                order_id=f"ext_buy_{t}",
                side=Side.BUY,
                price=mid + random.uniform(0.05, 0.20),  # near/above mid -> likely marketable vs asks
                quantity=random.choice([4, 6, 8, 10]),
                timestamp=t + 0.20,
            ))

        mm.step(timestamp=float(t))

    os.makedirs("data", exist_ok=True)
    with open("data/history_rule_based.json", "w") as f:
        json.dump([s.__dict__ for s in mm.get_history()], f, indent=2)

    last = mm.get_history()[-1]
    print(f"Steps={len(mm.get_history())}  Final PnL={last.pnl:.2f}  Final Inv={last.inventory}  Last mid={last.mid_price:.2f}")
    print("Wrote data/history_rule_based.json")
