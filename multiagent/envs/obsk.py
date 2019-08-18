import numpy as np

class Node():
    def __init__(self, label, qpos_ids, qvel_ids):
        self.label = label
        self.qpos_ids = qpos_ids
        self.qvel_ids = qvel_ids
        pass

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.label


class Edge():
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

    def __contains__(self, item):
        return (self.node1 is item) or (self.node2 is item)

    def __str__(self):
        return "Edge({},{})".format(self.node1, self.node2)

    def __repr__(self):
        return "Edge({},{})".format(self.node1, self.node2)


def get_joints_at_kdist(agent_id, agent_partitions, edges, k=0, kagents=False,):
    """ Identify all joints at distance <= k from agent agent_id

    :param agent_id: id of agent to be considered
    :param agent_partitions: list of joint tuples in order of agentids
    :param edges: list of tuples (joint1, joint2)
    :param k: kth degree
    :param kagents: True (observe all joints of an agent if a single one is) or False (individual joint granularity)
    :return:
        dict with k as key, and list of joints at that distance
    """
    assert not kagents, "kagents not implemented!"

    agent_joints = agent_partitions[agent_id]

    def _adjacent(lst, kagents=False):
        # return all sets adjacent to any element in lst
        ret = set([])
        for l in lst:
            ret = ret.union(set([(e.node1 if l is e.node2 else e.node2) for e in edges if l in e]))
        return ret

    seen = set([])
    new = set([])
    k_dict = {}
    for _k in range(k+1):
        if not _k:
            new = set(agent_joints)
        else:
            new = _adjacent(new) - seen
        seen = seen.union(new)
        k_dict[_k] = sorted(list(new), key=lambda x:x.label)
    return k_dict


def build_obs(k_dict, qpos, qvel, vec_len=None):
    """Given a k_dict from get_joints_at_kdist, extract observation vector.

    :param k_dict: k_dict
    :param qpos: qpos numpy array
    :param qvel: qvel numpy array
    :param vec_len: if None no padding, else zero-pad to vec_len
    :return:
    observation vector
    """
    obs_qpos_lst = []
    obs_qvel_lst = []
    for k in sorted(list(k_dict.keys())):
        for _t in k_dict[k]:
            obs_qpos_lst.append(qpos[_t.qpos_ids])
            obs_qvel_lst.append(qvel[_t.qvel_ids])

    ret = np.concatenate([obs_qpos_lst,
                          obs_qvel_lst])
    if vec_len is not None:
        pad = np.array((vec_len - len(obs_qpos_lst) - len(obs_qvel_lst))*[0])
        return np.concatenate([ret, pad])
    return ret


def get_parts_and_edges(label, partitioning):
    if label == "half_cheetah":

        # define Mujoco graph
        bthigh = Node("bthigh", 3, 3)
        bshin = Node("bshin", 4, 4)
        bfoot = Node("bfoot", 5, 5)
        fthigh = Node("fthigh", 6, 6)
        fshin = Node("fshin", 7, 7)
        ffoot = Node("ffoot", 8, 8)

        edges = [Edge(bfoot, bshin),
                 Edge(bshin, bthigh),
                 Edge(bthigh, fthigh),
                 Edge(fthigh, fshin),
                 Edge(fshin, ffoot)]

        if partitioning == "3x3":
            parts = [(bfoot, bshin, bthigh),
                     (ffoot, fshin, fthigh)]
        elif partitioning == "6x1":
            parts = [(bfoot,), (bshin,), (bthigh,), (ffoot,), (fshin,), (fthigh,)]

        return parts, edges