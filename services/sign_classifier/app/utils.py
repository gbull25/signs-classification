import redis


def create_redis_pool():
    return redis.ConnectionPool(
        host='redis',
        port=5370,
        db=0,
        decode_responses=False
    )


pool = create_redis_pool()
