import numpy as np
import pandas as pd

from neuclease.dvid import *
from neuclease.util import tqdm_proxy

from neuclease import configure_default_logging
configure_default_logging()

# Copied from neuroglancer
# Link:
# https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B8e-9%2C%22m%22%5D%2C%22y%22:%5B8e-9%2C%22m%22%5D%2C%22z%22:%5B8e-9%2C%22m%22%5D%7D%2C%22position%22:%5B30201.9453125%2C37924.96875%2C54272%5D%2C%22crossSectionScale%22:3.575961205895026%2C%22projectionOrientation%22:%5B-0.29115599393844604%2C-0.05720524117350578%2C-0.1234726831316948%2C0.9469478726387024%5D%2C%22projectionScale%22:2653.1478170244723%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%7B%22url%22:%22precomputed://gs://flyem-vnc-2-26-213dba213ef26e094c16c860ae7f4be0/v3_emdata_clahe_xy/jpeg%22%2C%22subsources%22:%7B%22default%22:true%2C%22bounds%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22name%22:%22v3_emdata_clahe_xy/jpeg%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://vnc-v3-seg-3d2f1c08fd4720848061f77362dc6c17/rc5_wsexp%22%2C%22tab%22:%22source%22%2C%22name%22:%22new-supervoxels%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://vnc-v3-seg-3d2f1c08fd4720848061f77362dc6c17/rc4_wsexp%22%2C%22tab%22:%22source%22%2C%22name%22:%22old-supervoxels%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://vnc-v3-seg-3d2f1c08fd4720848061f77362dc6c17/rc4_wsexp_rsg32_16_sep_8_sep1e6%22%2C%22tab%22:%22source%22%2C%22name%22:%22old-agglo%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://vnc-v3-seg-3d2f1c08fd4720848061f77362dc6c17/mask%22%2C%22tab%22:%22source%22%2C%22segmentQuery%22:%22sneaky%20comment%20here:%201:oob%202:trachea%203:glia%204:cell%20bodies%205:neuropil%22%2C%22name%22:%22voxel-classes%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22precomputed://gs://flyem-vnc-roi-d5f392696f7a48e27f49fa1a9db5ee3b/roi%22%2C%22subsources%22:%7B%22default%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22name%22:%22roi%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://flyem-vnc-roi-d5f392696f7a48e27f49fa1a9db5ee3b/nBreak-v1%22%2C%22tab%22:%22source%22%2C%22name%22:%22nBreak-v1%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22annotation%22%2C%22source%22:%7B%22url%22:%22local://annotations%22%2C%22transform%22:%7B%22outputDimensions%22:%7B%22x%22:%5B8e-9%2C%22m%22%5D%2C%22y%22:%5B8e-9%2C%22m%22%5D%2C%22z%22:%5B8e-9%2C%22m%22%5D%7D%7D%7D%2C%22tool%22:%22annotateBoundingBox%22%2C%22annotations%22:%5B%7B%22pointA%22:%5B13312%2C31744%2C30184%5D%2C%22pointB%22:%5B14336%2C32768%2C31208%5D%2C%22type%22:%22axis_aligned_bounding_box%22%2C%22id%22:%22772a59939a8a65f95bb7f2e27dfe544a616ba15f%22%7D%2C%7B%22pointA%22:%5B20992%2C13312%2C7680%5D%2C%22pointB%22:%5B22016%2C14336%2C8704%5D%2C%22type%22:%22axis_aligned_bounding_box%22%2C%22id%22:%22772a59939a8a65f95bb7f2e27dfe544a616ba15d%22%7D%2C%7B%22pointA%22:%5B29696%2C37376%2C54272%5D%2C%22pointB%22:%5B30720%2C38400%2C55296%5D%2C%22type%22:%22axis_aligned_bounding_box%22%2C%22id%22:%22772a59939a8a65f95bb7f2e27dfe544a616ba15c%22%7D%5D%2C%22name%22:%22subvolume-boxes%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://vnc-v4/32nm_candidates/337348382fb%22%2C%22tab%22:%22source%22%2C%22name%22:%22337348382fb%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://vnc-v4/32nm_candidates/357353626fb%22%2C%22tab%22:%22source%22%2C%22name%22:%22357353626fb%22%2C%22visible%22:false%7D%5D%2C%22showSlices%22:false%2C%22prefetch%22:false%2C%22selectedLayer%22:%7B%22layer%22:%22357353626fb%22%2C%22visible%22:true%7D%2C%22layout%22:%224panel%22%2C%22selection%22:%7B%22layers%22:%7B%22subvolume-boxes%22:%7B%22annotationId%22:%22772a59939a8a65f95bb7f2e27dfe544a616ba15c%22%2C%22annotationSource%22:0%2C%22annotationSubsource%22:%22default%22%7D%7D%7D%7D

ng_ann = [
 {'pointA': [13312, 31744, 30184],
  'pointB': [14336, 32768, 31208],
  'type': 'axis_aligned_bounding_box',
  'id': '772a59939a8a65f95bb7f2e27dfe544a616ba15f'},
 {'pointA': [20992, 13312, 7680],
  'pointB': [22016, 14336, 8704],
  'type': 'axis_aligned_bounding_box',
  'id': '772a59939a8a65f95bb7f2e27dfe544a616ba15d'},
 {'pointA': [29696, 37376, 54272],
  'pointB': [30720, 38400, 55296],
  'type': 'axis_aligned_bounding_box',
  'id': '772a59939a8a65f95bb7f2e27dfe544a616ba15c'}
]

# Load boxes
ann_boxes_xyz = np.array([(a['pointA'], a['pointB']) for a in  ng_ann])
ann_boxes_zyx = ann_boxes_xyz[..., ::-1]
boxes_zyx = np.zeros_like(ann_boxes_zyx)
boxes_zyx[:, 0, :] = ann_boxes_zyx.min(axis=1)
boxes_zyx[:, 1, :] = ann_boxes_zyx.max(axis=1)
assert (boxes_zyx[:, 1] - boxes_zyx[:, 0] > 0).all()

# Read segmentation and sizes
vnc_seg = ('emdata4:8450', '75d3ddd2e9e143a38fa9cc9e7d55b3d1', 'segmentation')
dfs = []
segs = []
for box in tqdm_proxy(boxes_zyx):
    seg = fetch_labelmap_voxels_chunkwise(*vnc_seg, box, threads=8)
    vc = pd.Series(seg.reshape(-1)).value_counts()
    df = vc.rename('boxed_size').rename_axis('body').reset_index()
    df['full_size'] = fetch_sizes(*vnc_seg, df['body'], batch_size=100, threads=8).values
    segs.append(seg)
    dfs.append(df)

for box, df in zip(boxes_zyx, dfs):
    name = '-'.join(map(str, box[0, ::-1].tolist()))
    df = df.sort_values('boxed_size', ascending=False)
    df.to_csv(f'seg-box-stats-{name}.csv', df, index=False, header=True)
