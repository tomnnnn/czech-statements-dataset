async def track_progress(coro, counter, total, unit=""):
    result = await coro
    print(f"{next(counter)}/{total} {unit}s completed", end="\r", flush=True)

    return result


def find_by_id(id, data):
    for d in data:
        if d["id"] == id:
            return d
    return None
