from typing import Optional
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from backend.graph.builder import compile_graph


class GraphRuntime:
    def __init__(self, checkpoint_path: str = "checkpoints.db"):
        self.checkpoint_path = checkpoint_path
        self._checkpointer_cm = None
        self._checkpointer = None
        self.graph_app = None

    async def startup(self):
        if self.graph_app is not None:
            return
        self._checkpointer_cm = AsyncSqliteSaver.from_conn_string(self.checkpoint_path)
        self._checkpointer = await self._checkpointer_cm.__aenter__()
        self.graph_app = compile_graph(self._checkpointer)

    async def shutdown(self):
        if self._checkpointer_cm is not None:
            await self._checkpointer_cm.__aexit__(None, None, None)
        self._checkpointer_cm = None
        self._checkpointer = None
        self.graph_app = None

    def get_graph(self):
        if self.graph_app is None:
            raise RuntimeError("Graph runtime not initialized")
        return self.graph_app
