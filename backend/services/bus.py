from typing import Optional, List, Set
import asyncio

class MessageBus:
    """
    内部消息发布/订阅总线，支持多个 WebSocket 客户端同时监听进度。
    """
    def __init__(self):
        # 存储所有活跃的订阅者队列
        self.subscribers: Set[asyncio.Queue] = set()

    async def subscribe(self) -> asyncio.Queue:
        """
        创建一个新的订阅队列并返回。
        """
        queue = asyncio.Queue()
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        """
        移除指定的订阅队列。
        """
        if queue in self.subscribers:
            self.subscribers.remove(queue)

    async def broadcast(self, node_name: str, message: str, percentage: Optional[int] = None):
        """
        将消息广播给所有活跃的订阅者。
        """
        payload = {
            "type": "PROGRESS",
            "node": node_name,
            "payload": message,
            "percentage": percentage
        }
        # 并发推送给所有订阅者
        if self.subscribers:
            await asyncio.gather(*[q.put(payload) for q in self.subscribers])

# 全局单例
message_bus = MessageBus()

async def broadcast_progress(node_name: str, message: str, percentage: Optional[int] = None):
    """
    便捷函数，供各专家节点执行进度广播。
    """
    await message_bus.broadcast(node_name, message, percentage)
