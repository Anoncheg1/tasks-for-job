from redis.cluster import RedisCluster
from redis import Redis
from redis.cluster import ClusterNode
import redis.cluster
from redis.connection import Connection, ConnectionPool
from redis.retry import Retry
from redis.backoff import NoBackoff, ConstantBackoff
from fastapi import FastAPI, HTTPException
import logging
from pydantic import BaseModel

app = FastAPI()

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


SHARDS={1:((1,  10), (31, 40)),
        2:((11, 20), (41, 50)),
        3:((21, 30), (51, 60))}


# -- fix 127.0.0.1:0
import random

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
    except redis.exceptions.ConnectionError as e:
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


rs1 = []
nodes = []
for pport in (30001, 30002,30003,30004, 30005):

    cn = ClusterNode(
        'localhost',
        # '10.255.255.1',
        pport,
        # redis_connection=r
    )
    # print("r", cn.redis_connection.connection_pool.connection_kwargs)
    nodes.append(cn) # 2 seconds timeout

# -------------------- MANIN ------------------

rc = RedisCluster(
    startup_nodes=nodes, decode_responses=False,
    socket_connect_timeout=2,
    dynamic_startup_nodes=False,
    socket_timeout=1, retry_on_timeout=False,
    retry=Retry(ConstantBackoff(2), 0),
    # retry_on_error=[ConnectionRefusedError],
    # single_connection_client=True
)

# -- Insert items to SortedSets
# for id in [1,2,11,12]:
#     sh = find_shard_id(SHARDS, id)
#     rc.zadd(f"data_{sh}",{f"Name{id}":id})
#     # print(sh, id, )


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


def get_one(item_id: int)->dict:
    "ZRANGE key start stop"
    sh = find_shard_id(SHARDS, item_id)
    resp = rc.zrange(f"data_{sh}", item_id, item_id)
    logger.info(f"zrange data_{sh} {item_id} {resp}")
    res = {item_id: r[0].decode("utf-8") for r in resp}
    return res


def add_one(item_id: int, name: str) -> bool:
    "ZADD key start stop"
    sh = find_shard_id(SHARDS, item_id)
    n = rc.zadd(f"data_{sh}", {name: item_id})
    logger.info(f"zadd data_{sh} {item_id} {n}")
    return bool(n)


def remove_one(item_id: int) -> dict:
    "ZREM key member."
    sh = find_shard_id(SHARDS, item_id)
    resp = rc.zrange(f"data_{sh}", item_id, item_id)
    logger.info(f"zrange data_{sh} {item_id} {resp}")
    res = {}
    for r in resp:
        v = r[0].decode("utf-8")
        res[item_id] = v
        resp = rc.zrem(f"data_{sh}", v)
    logger.info(f"zrem data_{sh} {item_id} {resp}")
    return res


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
        raise HTTPException(status_code=404, detail="Book not found")


# Endpoint to DELETE a book by ID.
@app.delete("/names/{item_id}", response_model=Data)
async def delete_item(item_id: int):
    res = remove_one(item_id)
    if res:
        return res
    else:
        raise HTTPException(status_code=404, detail="Book not found")


# Endpoint to create a new book
@app.post("/names/", response_model=Data)
async def create_book(d: Data):
    if add_one(d.idn, d.name):
        return d
    else:
        raise HTTPException(status_code=409, detail="Item already exists")
