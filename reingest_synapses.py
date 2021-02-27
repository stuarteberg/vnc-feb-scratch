import os
os.environ['DVID_ADMIN_TOKEN'] = 'stanfordrockskalsucks'

from neuclease.dvid import create_instance, load_gary_psds, post_tbar_jsons, post_psd_jsons, post_sync, post_reload

from neuclease import configure_default_logging
configure_default_logging()

print("Loading file")
psd_df = load_gary_psds('/nrs/flyem/huangg/vnc/synapses_v1.p')

print("Creating instance")
new_root = ('emdata5.janelia.org:8400', '1ec355123bf94e588557a4568d26d258')
create_instance(*new_root, 'synapses-reingest', 'annotation')

print("Loading tbars")
post_tbar_jsons(*new_root, 'synapses-reingest', psd_df, merge_existing=False, processes=16)

print("Loading psds")
post_psd_jsons(*new_root, 'synapses-reingest', psd_df, merge_existing=True, processes=16)

print("Posting sync")
post_sync(*new_root, 'synapses-reingest', ['segmentation'])

#print("Posting reload")
#post_reload(*new_root, 'synapses-reingest')
