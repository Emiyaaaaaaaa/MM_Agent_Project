from typing import Optional, List, Set, Any, Dict
import asyncio
from backend.services.serialization import ensure_blocks, make_stage

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

    async def broadcast(
        self,
        node_name: str,
        message: str,
        percentage: Optional[int] = None,
        artifact_url: Optional[str] = None,
        artifact_urls: Optional[List[str]] = None,
        artifact_manifest: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        将消息广播给所有活跃的订阅者。
        """
        inner: Dict[str, Any] = {
            "blocks": ensure_blocks(message),
            "percentage": percentage,
            "stage": make_stage(f"progress_{str(node_name).lower()}", f"{node_name} progress"),
        }
        if artifact_url:
            inner["artifact_url"] = artifact_url
        if artifact_urls:
            inner["artifact_urls"] = artifact_urls
        if artifact_manifest:
            inner["artifact_manifest"] = artifact_manifest
        payload = {
            "type": "PROGRESS",
            "node": node_name,
            "payload": inner,
        }
        # 并发推送给所有订阅者
        if self.subscribers:
            await asyncio.gather(*[q.put(payload) for q in self.subscribers])

# 全局单例
message_bus = MessageBus()

async def broadcast_progress(
    node_name: str,
    message: str,
    percentage: Optional[int] = None,
    artifact_url: Optional[str] = None,
    artifact_urls: Optional[List[str]] = None,
    artifact_manifest: Optional[List[Dict[str, Any]]] = None,
):
    """
    便捷函数，供各专家节点执行进度广播。
    """
    await message_bus.broadcast(
        node_name,
        message,
        percentage,
        artifact_url=artifact_url,
        artifact_urls=artifact_urls,
        artifact_manifest=artifact_manifest,
    )
