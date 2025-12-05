import redis
def clear_redis():
    r = redis.Redis(host="localhost", port=6380, db=0, decode_responses=True)
    r.flushdb()
    return {"status": "Redis DB cleared"}
if __name__ == "__main__":
    print(clear_redis())