import random
from redis.cluster import RedisCluster
# from redis import Redis
from redis.cluster import ClusterNode
import redis.cluster
# from redis.connection import Connection, ConnectionPool
from redis.retry import Retry
from redis.backoff import ConstantBackoff
 # NoBackoff
from fastapi import FastAPI, HTTPException
from starlette.responses import RedirectResponse
import logging
from pydantic import BaseModel
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi import FastAPI, Response
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response


app = FastAPI(title="Redis Cluster",
              version="0.1",
              description="Implementation of sharding table by ID with Sorted Set type.")


logger = logging.getLogger('uvicorn.error')


def find_shard_id(SHARDS, value):
    # Flatten the ranges into a list of tuples (start, end, shard_id)
    ranges = [(start, end, shard_id) for shard_id, ranges_list in SHARDS.items() for start, end in ranges_list]

    # Sort the ranges by their start values
    ranges.sort(key=lambda x: x[0])

    # Perform a binary search on the sorted list
    left, right = 0, len(ranges) - 1
    while left <= right:
        mid = (left + right) // 2
        if ranges[mid][0] <= value <= ranges[mid][1]:
            return ranges[mid][2]
        elif value < ranges[mid][0]:
            right = mid - 1
        else:
            left = mid + 1

    # If no matching range is found, return None
    return None


SHARDS = {1: ((1,  10), (31, 40)),
          2: ((11, 20), (41, 50)),
          3: ((21, 30), (51, 60))}


# -- fix 127.0.0.1:0


def my_get_nodes(self):
    # with filter if port = 0, for CLUSTER SLOTS with disconnected node
    return list([v for v in self.nodes_manager.nodes_cache.values() if v.port != 0 ])


def my_get_random_node(self):
    # with filter if port = 0, for CLUSTER SLOTS with disconnected node
    l = list([v for v in self.nodes_manager.nodes_cache.values() if v.port != 0 ])
    return random.choice(l)


RedisCluster.get_nodes = my_get_nodes

RedisCluster.get_random_node = my_get_random_node

# --- fix  Problem command fails if one of master node don't reply.
from redis.exceptions import ConnectionError, TimeoutError
orig_ec = RedisCluster._execute_command


def my_execute_command(self, target_node, *args, **kwargs):
    try:
        return orig_ec(self, target_node, *args, **kwargs)
    except redis.exceptions.ConnectionError:
        return []


RedisCluster._execute_command = my_execute_command


# NodesManager.initialize = myinitialize
import redis.cluster

redis.cluster.REDIS_ALLOWED_KEYS += (
    "socket_connect_timeout",
    "socket_timeout",
    "retry_on_timeout",
    "retry",
    "retry_on_error",
    "single_connection_client")


# -------------------- MAIN ------------------
rc = None


def get_all_data()-> list[dict]:
    "KEYS '*'"
    keys = rc.keys(target_nodes=RedisCluster.ALL_NODES)
    res = []
    for k in sorted(keys):
        resp = rc.zrange(k, 0, '-1', withscores=True)
        logger.info(resp)
        for r in resp:
            res.append({int(r[1]): r[0].decode("utf-8")})
    return res


def get_one(item_id: int) -> dict | None:
    "ZRANGE key start stop"
    sh = find_shard_id(SHARDS, item_id)
    resp = rc.zrangebyscore(f"data_{sh}", item_id, item_id)
    logger.info(f"zrangebyscore data_{sh} {item_id} {resp}")
    res = {"idn": item_id}
    if resp:
        res.update({"name": r.decode("utf-8") for r in resp})
        logger.info(res)
        return res
    else:
        return None


def add_one(item_id: int, name: str) -> bool:
    "ZADD key start stop"
    sh = find_shard_id(SHARDS, item_id)
    n = rc.zadd(f"data_{sh}", {name: item_id})
    logger.info(f"zadd data_{sh} {item_id} {n}")
    return bool(n)


def remove_one(item_id: int) -> dict:
    "ZREM key member."
    r = get_one(item_id)  # for return only
    if r:
        sh = find_shard_id(SHARDS, item_id)
        resp = rc.zremrangebyscore(f"data_{sh}", item_id, item_id)
        logger.info(f"zremrangebyscore data_{sh} {item_id} {resp}")
    return r


# Pydantic model for data
class Data(BaseModel):
    idn: int
    name: str


@app.get("/names")
async def read_root(response_model=list[Data]):
    return get_all_data()


# Endpoint to GET a book by ID.
@app.get("/names/{item_id}", response_model=Data)
async def read_item(item_id: int):
    res = get_one(item_id)
    if res:
        return res
    else:
        raise HTTPException(status_code=404, detail="Item not found")


# Endpoint to DELETE a book by ID.
@app.delete("/names/{item_id}", response_model=Data)
async def delete_item(item_id: int):
    res = remove_one(item_id)
    if res:
        return res
    else:
        raise HTTPException(status_code=404, detail="Item not found")


# Endpoint to create a new book
@app.post("/names/", response_model=Data)
async def create_item(d: Data):
    if add_one(d.idn, d.name):
        return d
    else:
        raise HTTPException(status_code=409, detail="Item already exists")

@app.get("/")
async def root():
    response = RedirectResponse(url='/docs', status_code=302)
    return response

def main():
    nodes = []
    for pport in (30001, 30002,30003,30004, 30005):
        cn = ClusterNode(
            'localhost',
            # '10.255.255.1',
            pport,
            # redis_connection=r
        )
        nodes.append(cn)  # 2 seconds timeout

    rc = RedisCluster(
        startup_nodes=nodes, decode_responses=False,
        socket_connect_timeout=2,
        dynamic_startup_nodes=False,
        socket_timeout=1, retry_on_timeout=False,
        retry=Retry(ConstantBackoff(2), 0),
        # retry_on_error=[ConnectionRefusedError],
        # single_connection_client=True
    )


if __name__ == "__main__":
    # init rc variable
    main()
