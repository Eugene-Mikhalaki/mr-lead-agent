"""Entry point for `python -m mr_lead_agent`."""

import asyncio

from mr_lead_agent.main import cli

if __name__ == "__main__":
    asyncio.run(cli())
