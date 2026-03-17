import asyncio
import logging
import os
import signal
import sys

from agents.orchestrator.orchestrator_agent import Orchestrator

log_dir = os.path.expanduser("~/crypto-trader-data/data")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"{log_dir}/trading.log"),
    ],
)

logger = logging.getLogger(__name__)


async def main():
    orchestrator = Orchestrator()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(orchestrator)))

    await orchestrator.start()


async def shutdown(orchestrator: Orchestrator):
    logger.info("Shutdown signal received")
    await orchestrator.stop()
    asyncio.get_event_loop().stop()


if __name__ == "__main__":
    asyncio.run(main())
