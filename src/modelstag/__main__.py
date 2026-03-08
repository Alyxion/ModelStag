"""CLI entry point for ModelStag."""

import argparse
import asyncio
import sys
import json

import uvicorn


def cmd_serve(args):
    """Start the API server."""
    from modelstag.config.settings import get_settings

    # Override settings if needed
    if args.single_process:
        import os
        os.environ["MODELSTAG_PROCESS_MODE"] = "single"

    settings = get_settings()

    uvicorn.run(
        "modelstag.api.app:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
        log_level="info",
        workers=1 if args.reload else args.workers,
    )


def cmd_status(args):
    """Show status of all models."""
    from modelstag.config.settings import get_settings
    from modelstag.manager.pid_store import PidStore

    settings = get_settings()
    pid_store = PidStore(settings.pids_dir)

    print(f"Process mode: {settings.process_mode.value}")
    print(f"Config file: {settings.config_path}")
    print()

    # List configured models
    print("Configured models:")
    for config in settings.get_enabled_models():
        print(f"  - {config.name} ({config.type.value}, startup={config.startup.value})")

    print()

    # List running workers
    workers = pid_store.list_all()
    print(f"Running workers ({len(workers)}):")

    for info in workers:
        alive = "running" if info.is_alive() else "dead"
        print(f"  - {info.model_name}: PID {info.pid} ({alive})")

    if not workers:
        print("  (none)")


def cmd_stop_all(args):
    """Stop all running workers."""
    from modelstag.config.settings import get_settings
    from modelstag.manager.pid_store import PidStore
    import os
    import signal

    settings = get_settings()
    pid_store = PidStore(settings.pids_dir)

    workers = pid_store.list_all()
    print(f"Stopping {len(workers)} workers...")

    for info in workers:
        if info.is_alive():
            try:
                os.kill(info.pid, signal.SIGTERM)
                print(f"  Sent SIGTERM to {info.model_name} (PID {info.pid})")
            except Exception as e:
                print(f"  Failed to stop {info.model_name}: {e}")

    # Clean up
    pid_store.cleanup_stale()
    print("Done.")


def cmd_start_all(args):
    """Start all eager models."""
    from modelstag.config.settings import get_settings
    from modelstag.manager.process_manager import ProcessManager

    settings = get_settings()
    manager = ProcessManager(settings)

    async def run():
        await manager.startup()
        print("Started eager models. Use 'modelstag status' to check.")

    asyncio.run(run())


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="modelstag",
        description="Multi-model hosting system for AI vision models",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", type=str, help="Host to bind to")
    serve_parser.add_argument("--port", type=int, help="Port to bind to")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    serve_parser.add_argument(
        "--single-process", action="store_true", help="Run in single-process mode"
    )
    serve_parser.add_argument(
        "--workers", type=int, default=1, help="Number of uvicorn workers"
    )
    serve_parser.set_defaults(func=cmd_serve)

    # status command
    status_parser = subparsers.add_parser("status", help="Show status of all models")
    status_parser.set_defaults(func=cmd_status)

    # start-all command
    start_parser = subparsers.add_parser("start-all", help="Start all eager models")
    start_parser.set_defaults(func=cmd_start_all)

    # stop-all command
    stop_parser = subparsers.add_parser("stop-all", help="Stop all running workers")
    stop_parser.set_defaults(func=cmd_stop_all)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
