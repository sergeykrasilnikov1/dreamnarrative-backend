import json, tempfile, shutil
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from app.core.config import settings

api = KaggleApi()
api.authenticate()
script = Path('kaggle/sdxl_inference.py')
with tempfile.TemporaryDirectory() as tmpdir:
    folder = Path(tmpdir)
    shutil.copy(script, folder / 'sdxl_inference.py')
    meta = {
        'id': f'{settings.KAGGLE_USERNAME}/{settings.KAGGLE_KERNEL_SLUG}',
        'title': 'DreamNarrative SDXL Inference',
        'code_file': 'sdxl_inference.py',
        'language': 'python',
        'kernel_type': 'script',
        'is_private': True,
        'enable_gpu': True,
        'enable_internet': True,
        'dataset_sources': [f'{settings.KAGGLE_USERNAME}/dreamnarrative-payload'],
        'competition_sources': [], 'kernel_sources': [], 'model_sources': [],
    }
    (folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    api.kernels_push(str(folder))
    print('pushed')

import time
from app.services.kaggle_service import _get_kaggle_api, _kernel_ref, _fetch_kernel_log_error
api = _get_kaggle_api()
ref = _kernel_ref(settings.KAGGLE_KERNEL_SLUG)
for i in range(18):
    s = api.kernels_status(ref)
    state = s.get('status')
    print(f'poll {i}: {state}')
    if state == 'complete':
        print('SUCCESS')
        break
    if state == 'error':
        print('ERR log:', _fetch_kernel_log_error(api, settings.KAGGLE_KERNEL_SLUG)[:800])
        break
    if state == 'running':
        time.sleep(20)
    else:
        time.sleep(10)