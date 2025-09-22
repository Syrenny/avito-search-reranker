# cluster_run.py
import sys

from dask.distributed import Client, LocalCluster


def main():
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=True,
        memory_limit="9GB",
        scheduler_port=8786
    )
    client = Client(cluster)
    print("Dashboard:", cluster.dashboard_link)
    print("Workers:", len(client.scheduler_info().get("workers", {})))
    client.wait_for_workers(1)
    input("Press Enter to shut down...")  # держим процесс живым


if __name__ == "__main__":
    # На Linux/macOS можно принудительно включить 'fork' (Windows это игнорирует)
    try:
        import multiprocessing as mp

        if sys.platform != "win32":
            mp.set_start_method("fork", force=True)
    except Exception:
        pass

    main()
