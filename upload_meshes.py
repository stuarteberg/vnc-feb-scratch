import os
import sys
import glob
from tqdm import tqdm

from neuclease.util import iter_batches
from neuclease.dvid import *

from neuclease import configure_default_logging
configure_default_logging()

SERVER = 'emdata5.janelia.org:8400'

MESH_KV = 'segmentation_meshes'

UUID = find_master(SERVER)
#UUID = '93f1c3e4e53c40b4a84c20768c3990da'
#UUID = '3d55ca9512c946e0be79575cb06b8bd0'
print(f"uploading to {UUID}")

PROCESSES = 16

MESH_DIR = sys.argv[1]

os.chdir(MESH_DIR)

mesh_paths = sorted(glob.glob('*.ngmesh'))

def upload_mesh(p):
    post_key(SERVER, UUID, MESH_KV, p, open(p, 'rb').read())

compute_parallel(upload_mesh, mesh_paths, processes=PROCESSES, ordered=False)

#batches = iter_batches(mesh_paths, 100)
#for batch in tqdm(batches):
#    meshes = {p: open(p, 'rb').read() for p in batch}
#    post_keyvalues(SERVER, UUID, MESH_KV, meshes)

print(f"Loaded {len(mesh_paths)} meshes.")
