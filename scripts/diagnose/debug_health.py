import asyncio
from backend.main import health_check

async def main():
    try:
        res = await health_check()
        print(res)
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(main())
