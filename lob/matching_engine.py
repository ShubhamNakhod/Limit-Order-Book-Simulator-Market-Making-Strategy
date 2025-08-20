# lob/matching_engine.py

from enum import Enum
import heapq
from typing import NamedTuple, List, Dict, Tuple, Optional


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


class Order(NamedTuple):
    order_id: str
    side: Side
    price: float
    quantity: int
    timestamp: float  # lower = earlier arrival (FIFO at same price)


class Trade(NamedTuple):
    price: float
    quantity: int
    buy_order_id: str
    sell_order_id: str


HeapEntry = Tuple[float, float, str]  # (price or -price, timestamp, order_id)


class OrderBook:
    """
    Price–time priority LOB using two heaps:
      - bids: max-heap via negative price  (-price, ts, id)
      - asks: min-heap via positive price ( price, ts, id)

    Active orders live in order_map; cancelled/filled orders are lazily
    removed when they reach the top of the heap.
    """

    def __init__(self) -> None:
        self.bids: List[HeapEntry] = []
        self.asks: List[HeapEntry] = []
        self.order_map: Dict[str, Order] = {}

    # --------------------------- utilities ---------------------------

    def _clean_top(self, heap: List[HeapEntry]) -> None:
        """Pop heap until the top entry refers to a live order (lazy deletion)."""
        while heap:
            _, _, oid = heap[0]
            if oid in self.order_map:
                return
            heapq.heappop(heap)

    def _best_ask(self) -> Optional[Order]:
        self._clean_top(self.asks)
        if not self.asks:
            return None
        _, _, oid = self.asks[0]
        return self.order_map.get(oid)

    def _best_bid(self) -> Optional[Order]:
        self._clean_top(self.bids)
        if not self.bids:
            return None
        _, _, oid = self.bids[0]
        return self.order_map.get(oid)

    # --------------------------- core API ---------------------------

    def insert_order(self, order: Order) -> List[Trade]:
        """
        Match the incoming order against the opposite side, then rest any
        remaining quantity on its own side. Always returns a list of Trade.
        """
        trades: List[Trade] = []
        incoming = order

        if incoming.side == Side.BUY:
            # Cross against best asks while we can
            while incoming.quantity > 0:
                best_ask = self._best_ask()
                if best_ask is None or incoming.price < best_ask.price:
                    break

                # Execute at resting order's price
                qty = min(incoming.quantity, best_ask.quantity)
                trades.append(
                    Trade(
                        price=best_ask.price,
                        quantity=qty,
                        buy_order_id=incoming.order_id,
                        sell_order_id=best_ask.order_id,
                    )
                )

                # Update both orders
                incoming = incoming._replace(quantity=incoming.quantity - qty)
                remaining = best_ask.quantity - qty

                # Pop top ask and reinsert if partially filled
                heapq.heappop(self.asks)
                if remaining > 0:
                    updated = best_ask._replace(quantity=remaining)
                    self.order_map[best_ask.order_id] = updated
                    heapq.heappush(self.asks, (updated.price, updated.timestamp, updated.order_id))
                else:
                    # fully filled
                    self.order_map.pop(best_ask.order_id, None)

            # Rest remaining BUY quantity
            if incoming.quantity > 0:
                self.order_map[incoming.order_id] = incoming
                heapq.heappush(self.bids, (-incoming.price, incoming.timestamp, incoming.order_id))

        else:  # SELL
            # Cross against best bids while we can
            while incoming.quantity > 0:
                best_bid = self._best_bid()
                if best_bid is None or incoming.price > best_bid.price:
                    break

                qty = min(incoming.quantity, best_bid.quantity)
                trades.append(
                    Trade(
                        price=best_bid.price,
                        quantity=qty,
                        buy_order_id=best_bid.order_id,
                        sell_order_id=incoming.order_id,
                    )
                )

                incoming = incoming._replace(quantity=incoming.quantity - qty)
                remaining = best_bid.quantity - qty

                heapq.heappop(self.bids)
                if remaining > 0:
                    updated = best_bid._replace(quantity=remaining)
                    self.order_map[best_bid.order_id] = updated
                    heapq.heappush(self.bids, (-updated.price, updated.timestamp, updated.order_id))
                else:
                    self.order_map.pop(best_bid.order_id, None)

            # Rest remaining SELL quantity
            if incoming.quantity > 0:
                self.order_map[incoming.order_id] = incoming
                heapq.heappush(self.asks, (incoming.price, incoming.timestamp, incoming.order_id))

        return trades

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order; removal from heaps is lazy."""
        if order_id in self.order_map:
            del self.order_map[order_id]
            return True
        return False
