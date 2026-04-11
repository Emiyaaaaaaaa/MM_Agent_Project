import asyncio
import json
import websockets

async def test_ws():
    uri = "ws://localhost:8000/ws/chat/test_user_789"
    async with websockets.connect(uri) as websocket:
        print(f"[Client] 已连接至 {uri}")
        
        # 发送测试消息
        msg = {
            "message": "帮我分析一下刚才上传的赛题并生成建模思路。",
            "thread_id": "test_thread_001"
        }
        await websocket.send(json.dumps(msg))
        print(f"[Client] 已发送消息: {msg['message']}")
        
        # 持续接收流式反馈
        print("\n[Client] 正在接收流式反馈...\n" + "="*50)
        try:
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                
                msg_type = data.get("type")
                node = data.get("node", "System")
                payload = data.get("payload", "")
                
                if msg_type == "TOKEN":
                    print(payload, end="", flush=True)
                elif msg_type == "PROGRESS":
                    print(f"\n[PROGRESS] {node}: {payload} ({data.get('percentage')}%)")
                elif msg_type == "STATUS":
                    print(f"\n[STATUS] {node}: {payload}")
                elif msg_type == "FINAL":
                    print(f"\n\n[FINAL] 任务已完成。")
                    break
        except Exception as e:
            print(f"\n[Client] 连接断开: {e}")

if __name__ == "__main__":
    # 请确保 FastAPI 已在 8000 端口运行
    # uvicorn backend.main:app --reload
    print("提示：请先确保后端服务已启动！")
    try:
         asyncio.run(test_ws())
    except Exception as e:
         print(f"无法连接到后端: {e}")
