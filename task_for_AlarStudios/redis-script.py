from redis.cluster import RedisCluster
from redis import Redis
from redis.cluster import ClusterNode
import redis.cluster
from redis.connection import Connection, ConnectionPool
from redis.retry import Retry
from redis.backoff import NoBackoff, ConstantBackoff

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






# -- monkey patch



# from redis.connection import Connection

# def myf(self):
#     "Connection."
#     print("Connection------")
#     print("Connection, socket_connect_timeout", self.socket_connect_timeout)
#     print("Connection, socket_timeout", self.socket_timeout)
#     print(self.socket_connect_timeout,
#           self.socket_timeout,
#           self.retry_on_timeout,
#           self.retry,
#           self.retry_on_error)
#     # print("Connection, available_connections", self._available_connections)
#     print(self.retry._backoff)
#     print(self.host, self.port, self.socket_type)
#     # self.retry._supported_errors += (ConnectionRefusedError,)
#     sock = self.retry.call_with_retry(
#         lambda: self._connect(), lambda error: self.disconnect(error)
#     )

#     print("socks", sock)


#     print(self.retry._supported_errors)
#     print("_retries", self.retry._retries)
#     print("-------")
#     self._sock = sock
#     if self.redis_connect_func is None:
#         # Use the default on_connect function
#         self.on_connect()
#         print("here")
#     else:
#         # Use the passed function redis_connect_func
#         self.redis_connect_func(self)
#         print("here2")


# Connection.connect = myf

# def makec(self):
#     "Create a new connection"
#     print("ConnectionPool.make_connection", self.connection_kwargs)
#     if self._created_connections >= self.max_connections:
#         raise ConnectionError("Too many connections")
#     self._created_connections += 1
#     return self.connection_class(**self.connection_kwargs)

# ConnectionPool.make_connection = makec

# orig = ConnectionPool.get_connection

# def getc(self, command_name: str, *keys, **options):
#     "Create a new connection"
#     print("ConnectionPool, available_connections", keys, options)
#     return orig(self, command_name, keys, options)

# ConnectionPool.get_connection = getc

# from redis.retry import Retry

# from time import sleep

# def myf2(self, do, fail):
#     "Retry"
#     self._backoff.reset()
#     failures = 0
#     # print(self._supported_errors)
#     print("retry--------")
#     while True:
#         # print(failures)
#         try:
#             return do()
#         except self._supported_errors as error:
#             print("wtf")
#             failures += 1
#             fail(error)
#             print("wtf2", self._retries, failures, self._retries)
#             if self._retries >= 0 and failures > self._retries:
#                 raise error
#             backoff = self._backoff.compute(failures)
#             print("wtf3", backoff)
#             if backoff > 0:
#                 sleep(backoff)

# Retry.call_with_retry = myf2


# # from redis.cluster import NodesManager

# def myget_redis_connection(self, node):
#     print("myget_redis", node, node.redis_connection.connection_pool.connection_kwargs)
#     if not node.redis_connection:
#         with self._lock:
#             if not node.redis_connection:
#                 self.nodes_manager.create_redis_connections([node])
#     return node.redis_connection

# RedisCluster.get_redis_connection = myget_redis_connection

# def value_getter(self):
#     return self._value
# def value_setter(self, value):
#     print(f"Value changed {value}")
#     self._value = value

# # Monkey patch the ExistingClass to add the property setter
# ConnectionPool._available_connections = property(value_getter, value_setter)


# def myvget_connection(redis_node, *args, **options):
#     print("myvget_connection", redis_node.connection_pool.connection_kwargs)
#     return redis_node.connection or redis_node.connection_pool.get_connection(
#         args[0], **options
#     )

# redis.cluster.get_connection = myvget_connection

# orig = RedisCluster._execute_command

# def my_execute_command(self, target_node, *args, **kwargs):
#     print("HAY", target_node.redis_connection.connection_pool.connection_kwargs)
#     return orig(self, target_node, *args, **kwargs)

# RedisCluster._execute_command = my_execute_command


# orig = RedisCluster._determine_nodes

# def my_determine_nodes(self, *args, **kwargs):
#     v = orig(self, *args, **kwargs)
#     print("HAY", len(v), v[0].redis_connection.connection_pool.connection_kwargs)

#     return v

# RedisCluster._determine_nodes = my_determine_nodes



# orig = NodesManager._get_or_create_cluster_node

# def my_update_moved_slots(self, host, port, role, tmp_nodes_cache):
#     node_name = f"{host}:{port}"
#     target_node = tmp_nodes_cache.get(node_name)
#     v = orig(self, host, port, role, tmp_nodes_cache)
#     print("here", target_node)
#     return(v)

# NodesManager._get_or_create_cluster_node = my_update_moved_slots


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


# orig_ec = RedisCluster._execute_command

# def my_execute_command(self, target_node, *args, **kwargs):
#     print(self.get_nodes())
#     return orig_ec(self, target_node, *args, **kwargs)

# RedisCluster._execute_command = my_execute_command

# --- fix  Problem command fails if one of master node don't reply.
from redis.exceptions import ConnectionError, TimeoutError
orig_ec = RedisCluster._execute_command

def my_execute_command(self, target_node, *args, **kwargs):
    try:
        return orig_ec(self, target_node, *args, **kwargs)
    except redis.exceptions.ConnectionError as e:
        return []

RedisCluster._execute_command = my_execute_command

# --- test timeout
import time
orig_connect = Connection.connect

def my_connect(self):
    start_time = time.time()
    orig_connect(self)
    end_time = time.time()
    print(f"Connect executed in {end_time - start_time:.2f} seconds")


Connection.connect = my_connect


orig_getc = ConnectionPool.get_connection


def getc(self, command_name: str, *keys, **options):
    "Create a new connection"
    start_time = time.time()
    try:
        c = orig_getc(self, command_name, keys, options)
    except Exception as e:
        print(e)
        raise e
    finally:
        end_time = time.time()
        print(f"Connect executed in {end_time - start_time:.2f} seconds")

    return c

ConnectionPool.get_connection = getc

# import socket
# orig_sock = socket.socket
# def myconnect(family, socktype, proto):
#     time.sleep(1)
#     return orig_sock(family, socktype, proto)
# socket.socket = myconnect

# from redis.cluster import get_node_name

# def my_get_or_create_cluster_node(self, host, port, role, tmp_nodes_cache):
#     print("wtf222")
#     node_name = get_node_name(host, port)
#     # check if startup node exist to get redis_connection from it
#     snode = self.startup_nodes.get(get_node_name(host, port))
#     snode_rc = snode.redis_connection if snode else None
#     # check if we already have this node in the tmp_nodes_cache
#     target_node = tmp_nodes_cache.get(node_name)
#     if target_node is None:
#         # before creating a new cluster node, check if the cluster node already
#         # exists in the current nodes cache and has a valid connection so we can
#         # reuse it
#         target_node = self.nodes_cache.get(node_name)
#         if target_node is None or target_node.redis_connection is None:
#             # if snode_rc is None:

#             # create new cluster node for this cluster
#             target_node = ClusterNode(host, port, role,
#                                       redis_connection=snode_rc)
#         if target_node.server_type != role:
#             target_node.server_type = role
#     return target_node

# NodesManager._get_or_create_cluster_node = my_get_or_create_cluster_node


# def mycreate_redis_node(self, host, port, **kwargs):
#     print("mycreate_redis_node", kwargs)
#     # print(REDIS_ALLOWED_KEYS)
#     if self.from_url:
#         # Create a redis node with a costumed connection pool
#         kwargs.update({"host": host})
#         kwargs.update({"port": port})
#         r = Redis(connection_pool=self.connection_pool_class(**kwargs))
#     else:
#         r = Redis(host=host, port=port, **kwargs)
#     return r

# NodesManager.create_redis_node = mycreate_redis_node

# orig_initialize = NodesManager.initialize

# def myinitialize(self):
#     print("myinitit", self.connection_kwargs)
#     return orig_initialize(self)

# NodesManager.initialize = myinitialize
import redis.cluster

redis.cluster.REDIS_ALLOWED_KEYS += (
    "socket_connect_timeout",
    "socket_timeout",
    "retry_on_timeout",
    "retry",
    "retry_on_error",
    "single_connection_client")

# orig_clean = redis.cluster.cleanup_kwargs

# MY_REDIS_ALLOWED_KEYS = REDIS_ALLOWED_KEYS + (
#     "socket_connect_timeout",
#     "socket_timeout",
#     "retry_on_timeout",
#     "retry",
#     "retry_on_error",
#     "single_connection_client")

# def mycleanup_kwargs(**kwargs):
#     connection_kwargs = {
#         k: v
#         for k, v in kwargs.items()
#         if k in MY_REDIS_ALLOWED_KEYS and k not in KWARGS_DISABLED_KEYS
#     }
#     return connection_kwargs

# redis.cluster.cleanup_kwargs = mycleanup_kwargs


rs1 = []
nodes = []
for pport in (30001, 30002,30003,30004, 30005):

    # print("r", r.connection_pool.connection_kwargs)
    # if pport < 30002:
        # try:
        # r = Redis(
        # # 'localhost',
        # # '127.0.0.1',
        # '10.255.255.1',
        # pport,
        # socket_connect_timeout=2,
        # socket_timeout=2, retry_on_timeout=False,
        # retry=Retry(ConstantBackoff(2), 0),
        # retry_on_error=[ConnectionRefusedError, TimeoutError],
        # # single_connection_client=True
        # )
    # except:

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
    startup_nodes=nodes, # decode_responses=True,
    socket_connect_timeout=2,
    dynamic_startup_nodes=False,
    socket_timeout=1, retry_on_timeout=False,
    retry=Retry(ConstantBackoff(2), 0),
    # retry_on_error=[ConnectionRefusedError],
    # single_connection_client=True
)
rs2 = []
# print("wtf11")
# [rs2.append(v.redis_connection) for k,v in rc.nodes_manager.startup_nodes.items()]
# [print ("rs1rs2", id(x.connection_pool), id(v.connection_pool)) for x, v in zip(rs1, rs2)]
# print(rc.nodes_manager.startup_nodes)
print("''''''''''")
# [print(k, v.redis_connection.connection_pool.connection_kwargs) for k,v in rc.nodes_manager.startup_nodes.items()]
# [print(k, v.redis_connection.connection_pool.connection_kwargs) for k,v in rc.nodes_manager.startup_nodes.items()]
# print(rc.nodes_manager.startup_nodes)
# [print(k, v.nodes_manager.connection_kwargs) for k,v in rc.nodes_manager.startup_nodes.items()]

# [print(k, v.redis_connection.connection_pool._available_connections) for k,v in rc.nodes_manager.startup_nodes.items()]
# print("_________________")
# [print("hh", k, bool(v.redis_connection.info().get("cluster_enabled"))) for k,v in rc.nodes_manager.startup_nodes.items()]
# [print(k, v) for k,v in rc.nodes_manager.startup_nodes.items()]
print("startup_nodes:")
[print (x) for x in rc.nodes_manager.startup_nodes]
print()
print("nodes_cache:")
[print (x) for x in rc.nodes_manager.nodes_cache]
print()
print("read argiments:")
import sys
import re
print ('Argument List:', str(sys.argv))

args = []
for a in sys.argv[1:]:
    if re.match("^[0-9]*$", a):
        args.append(int(a))
    else:
        args.append(a)
rc_command = getattr(rc, args[0])
# print(args[1:])

# -- call command with arguments
if args[0] == "zadd":
    print(rc_command(args[1], {args[2]:args[3]}))

else:
    print(rc_command(*args[1:]))
# print(rc.ping(target_nodes=RedisCluster.RANDOM))
# -- set:
# sh = 1
# id = 1
# name = 'Test 1'
# print(rc.set(f"data_1:{id}:{{{sh}}}", "Name1"))

# sh = 2
# id = 12
# name = 'Test 2'
# print(rc.set(f"data_1:{id}:{{{sh}}}", "Name2"))

# -- get one:
id = 1
# sh = find_shard_id(SHARDS, id)
# print(rc.get(f"data_1:{id}:{{{sh}}}"))


# [print(x) for x in rc.nodes_manager.startup_nodes]

# # -- get all:
print(rc.keys(target_nodes=RedisCluster.ALL_NODES))
# print(rc.get("data_1:1:{11}"))
print(rc.get("data_1:1:{11}"))
# print(rc.keys(target_nodes=RedisCluster.))


# -- get range (TODO):
# Sorted set at each shard + "get all"
