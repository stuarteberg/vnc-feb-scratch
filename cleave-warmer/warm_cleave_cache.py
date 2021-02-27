import sys
import requests

import numpy as np
import pandas as pd

from io import BytesIO
from tqdm import tqdm

import pandas as pd

from neuclease.util import compute_parallel

from neuclease import configure_default_logging
configure_default_logging()

USER = "bergs"
CLEAVE_SERVER = "emdata4.int.janelia.org:5590"
DVID_SERVER = "emdata5-private.janelia.org:8400"
DVID_UUID = "f48d2a0b298b46198a2170900b2462d7"
THREADS = 4

BODY_CSV = sys.argv[1]

def fetch_body_edge_table(cleave_server, dvid_server, uuid, instance, body):
    dvid_server, dvid_port = dvid_server.split(':')

    if not cleave_server.startswith('http'):
        cleave_server = 'http://' + cleave_server

    data = { "body-id": body,
             "port": dvid_port,
             "server": dvid_server,
             "uuid": uuid,
             "segmentation-instance": instance,
             "user": "cache-warmer" }

    r = requests.post(f'{cleave_server}/body-edge-table', json=data)
    r.raise_for_status()

    df = pd.read_csv(BytesIO(r.content), header=0)
    df = df.astype({'id_a': np.uint64, 'id_b': np.uint64, 'score': np.float32})
    return df


def warm_body(body):
    fetch_body_edge_table(CLEAVE_SERVER, DVID_SERVER, DVID_UUID, 'segmentation', body)

bodies = pd.read_csv(BODY_CSV)['body']
_ = compute_parallel(warm_body, bodies, threads=THREADS, ordered=False)
