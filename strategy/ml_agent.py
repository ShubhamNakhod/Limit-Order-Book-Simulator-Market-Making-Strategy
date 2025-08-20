# strategy/ml_agent.py

import math
import uuid
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np

from lob.matching_engine import OrderBook, Order, Side, Trade


# ------------------------- Context & Bandit -------------------------

@dataclass
class Context:
    """
    State features observed by the agent at a decision time.
    """
    inventory: float            # normalized inventory in [-1, 1]
    imbalance: float            # order flow imbalance (placeholder here)
    volatility: float           # recent mid-price volatility
    mid_price: float            # current mid-price

    def to_vector(self) -> np.ndarray:
        # Bias + 3 features = 4-dim vector
        return np.array([1.0, self.inventory, self.imbalance, self.volatility], dtype=float)


class LinUCB:
    """
    Simple LinUCB contextual bandit for choosing a half-spread.
    Each action 'a' has its own linear model:
        A_a (dxd), b_a (dx1)
    We pick: argmax_a [ theta_a^T x + alpha * sqrt(x^T A_a^{-1} x) ]
    """
    def __init__(self, actions: List[float], alpha: float = 1.0):
        self.actions = actions
        self.alpha = alpha
        self.d = 4  # dimension of context vector
        self.A: Dict[float, np.ndarray] = {a: np.eye(self.d) for a in actions}
        self.b: Dict[float, np.ndarray] = {a: np.zeros(self.d, dtype=float) for a in actions}

    def select(self, context: Context) -> float:
        x = context.to_vector()
        best_score = -float("inf")
        best_action = self.actions[0]
        for a in self.actions:
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]                         # (d,)
            mean = float(theta @ x)                           # scalar
            exploration = self.alpha * math.sqrt(float(x @ A_inv @ x))
            score = mean + exploration
            if score > best_score:
                best_score = score
                best_action = a                               # <-- bugfix: assign action, not score
        return best_action

    def update(self, chosen_action: float, context: Context, reward: float) -> None:
        x = context.to_vector()
        self.A[chosen_action] += np.outer(x, x)               # dxd
        self.b[chosen_action] += reward * x                   # d


# ------------------------- ML Market Maker -------------------------

class MLMarketMaker:
    """
    Market maker that chooses the half-spread via LinUCB given a context.
    Still uses a simple linear inventory skew around mid-price.
    """
    def __init__(
        self,
        order_book: OrderBook,
        base_alpha: float = 0.05,
        order_size: int = 10,
        max_inventory: int = 100,
        spread_candidates: Optional[List[float]] = None,
        exploration_coef: float = 1.0,
        reward_inventory_penalty: float = 0.0,   # set >0 to penalize inventory
    ):
        if spread_candidates is None:
            spread_candidates = [0.2, 0.5, 1.0, 1.5]

        self.ob = order_book
        self.alpha = base_alpha
        self.order_size = order_size
        self.max_inventory = max_inventory
        self.bandit = LinUCB(actions=spread_candidates, alpha=exploration_coef)

        self.current_bid_id: Optional[str] = None
        self.current_ask_id: Optional[str] = None

        self.inventory = 0
        self.cash = 0.0
        self.last_mid = 0.0

        self.prev_pnl: Optional[float] = None
        self.reward_inventory_penalty = reward_inventory_penalty

        self.history: List[Dict[str, Any]] = []  # logs for analysis/dashboard

    # ---------- book helpers ----------

    def _compute_mid(self) -> float:
        """
        Compute mid from top-of-book; lazily cleans heaps if needed.
        Falls back to the last seen mid if book is empty on one or both sides.
        """
        import heapq

        best_bid = None
        best_ask = None

        # Clean/peek bid
        while self.ob.bids:
            neg_price, ts, oid = self.ob.bids[0]
            if oid in self.ob.order_map:
                best_bid = -neg_price
                break
            heapq.heappop(self.ob.bids)

        # Clean/peek ask
        while self.ob.asks:
            price, ts, oid = self.ob.asks[0]
            if oid in self.ob.order_map:
                best_ask = price
                break
            heapq.heappop(self.ob.asks)

        if best_bid is not None and best_ask is not None:
            mid = 0.5 * (best_bid + best_ask)
        elif best_bid is not None:
            mid = best_bid + 0.5
        elif best_ask is not None:
            mid = best_ask - 0.5
        else:
            mid = self.last_mid

        if mid:
            self.last_mid = mid
        return self.last_mid

    def _compute_imbalance(self) -> float:
        """
        Placeholder: return 0.0 (neutral). Replace with true order-flow imbalance later.
        """
        return 0.0

    @staticmethod
    def _compute_volatility(recent_mid_prices: List[float]) -> float:
        if len(recent_mid_prices) < 2:
            return 0.0
        arr = np.asarray(recent_mid_prices, dtype=float)
        rets = np.diff(np.log(arr))
        return float(np.std(rets))

    def _cancel_existing_quotes(self) -> None:
        if self.current_bid_id:
            self.ob.cancel_order(self.current_bid_id)
            self.current_bid_id = None
        if self.current_ask_id:
            self.ob.cancel_order(self.current_ask_id)
            self.current_ask_id = None

    # ---------- core step ----------

    def step(self, timestamp: float, recent_mid_prices: List[float]) -> None:
        """
        One decision step:
          1) cancel stale quotes
          2) compute context
          3) select half-spread
          4) place inventory-skewed quotes
          5) update inventory/cash from fills
          6) compute reward and update bandit
          7) log state
        """
        # 1) cancel
        self._cancel_existing_quotes()

        # 2) context
        mid = self._compute_mid()
        imbalance = self._compute_imbalance()
        volatility = self._compute_volatility(recent_mid_prices)
        norm_inventory = max(-self.max_inventory, min(self.inventory, self.max_inventory)) / self.max_inventory

        ctx = Context(
            inventory=norm_inventory,
            imbalance=imbalance,
            volatility=volatility,
            mid_price=mid,
        )

        # 3) select spread
        chosen_spread = self.bandit.select(ctx)

        # 4) inventory-skewed quotes
        skew = self.alpha * self.inventory
        bid_price = mid - chosen_spread + skew
        ask_price = mid + chosen_spread + skew

        # post bid
        bid_id = str(uuid.uuid4())
        bid_order = Order(order_id=bid_id, side=Side.BUY, price=bid_price, quantity=self.order_size, timestamp=timestamp)
        self.current_bid_id = bid_id
        bid_trades = self.ob.insert_order(bid_order) or []     # guard against None

        # post ask
        ask_id = str(uuid.uuid4())
        ask_order = Order(order_id=ask_id, side=Side.SELL, price=ask_price, quantity=self.order_size, timestamp=timestamp)
        self.current_ask_id = ask_id
        ask_trades = self.ob.insert_order(ask_order) or []

        # 5) fills -> state update
        self._process_trades(bid_trades)
        self._process_trades(ask_trades)

        # 6) reward & bandit update
        new_mid = self._compute_mid()
        pnl = self.cash + self.inventory * new_mid

        # Reward: PnL delta minus (optional) inventory penalty
        if self.prev_pnl is None:
            reward = 0.0
        else:
            reward = pnl - self.prev_pnl
        reward -= self.reward_inventory_penalty * abs(self.inventory)

        self.bandit.update(chosen_spread, ctx, reward)
        self.prev_pnl = pnl

        # 7) log
        self.history.append({
            "timestamp": timestamp,
            "spread": chosen_spread,
            "inventory": self.inventory,
            "cash": self.cash,
            "mid_price": new_mid,
            "pnl": pnl,
            "ctx": {
                "inventory": ctx.inventory,
                "imbalance": ctx.imbalance,
                "volatility": ctx.volatility,
                "mid_price": ctx.mid_price,
            },
            "reward": reward,
        })

    # ---------- utils ----------

    def _process_trades(self, trades: Optional[List[Trade]]) -> None:
        if not trades:
            return
        for tr in trades:
            if tr.buy_order_id == self.current_bid_id:
                # we bought -> +inventory, -cash
                self.inventory += tr.quantity
                self.cash -= tr.price * tr.quantity
            elif tr.sell_order_id == self.current_ask_id:
                # we sold -> -inventory, +cash
                self.inventory -= tr.quantity
                self.cash += tr.price * tr.quantity

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history


# ------------------------- Demo Runner -------------------------

if __name__ == "__main__":
    """
    Minimal demo to produce history for the dashboard:
      - Seeds some passive liquidity so the mid has meaning
      - Runs the ML market maker for 50 steps
      - Writes data/history_ml.json
    """
    import os
    import json
    import random

    ob = OrderBook()
    mm = MLMarketMaker(
        order_book=ob,
        base_alpha=0.05,
        order_size=5,
        max_inventory=50,
        spread_candidates=[0.2, 0.5, 1.0, 1.5],
        exploration_coef=1.0,
        reward_inventory_penalty=0.0,
    )

    # seed passive book
    ob.insert_order(Order(order_id="ext_bid", side=Side.BUY,  price=99.5, quantity=100, timestamp=0))
    ob.insert_order(Order(order_id="ext_ask", side=Side.SELL, price=100.5, quantity=100, timestamp=0))

    recent_mids: List[float] = []

    for t in range(1, 51):
        # inject some random 'market pressure' orders
        if random.random() < 0.3:
            ob.insert_order(Order(order_id=f"mkt_sell_{t}", side=Side.SELL, price=99.0, quantity=10, timestamp=t + 0.1))
        if random.random() < 0.3:
            ob.insert_order(Order(order_id=f"mkt_buy_{t}",  side=Side.BUY,  price=101.0, quantity=10, timestamp=t + 0.2))

        # roll recent mids (for volatility calc)
        mid_now = mm._compute_mid()
        recent_mids.append(mid_now)
        recent_mids = recent_mids[-20:]  # keep a small window

        mm.step(timestamp=float(t), recent_mid_prices=recent_mids)

    os.makedirs("data", exist_ok=True)
    with open("data/history_ml.json", "w") as f:
        json.dump(mm.get_history(), f, indent=2)

    print("Wrote data/history_ml.json")
