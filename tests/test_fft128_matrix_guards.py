import copy
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('fft_matrix', Path(__file__).resolve().parents[1]/'scripts/run_fft128_cin240_matrix.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def sample():
    row = {'semantic': [{'index': 133, 'generated_sha256': 'fixed'}], 'audio_seconds': 5.62,
           'pcm_sha256': m.PCM['fft128'], 'comparison': {'same_shape': True, 'same_rate': True, 'max_pcm_delta': 1, 'snr_db': 92.4}}
    data = {k: True for k in ('checkpoint_files_unchanged', 'parameters_unchanged', 'ar_parameters_unchanged', 'student_files_unchanged')}
    data.update(cgroup={'cpu.max': '200000 100000', 'memory.max': '8000000000', 'memory.swap.max': '0'},
                approximate_ar_int8_only=True, experimental_options={'fast_experimental': False}, fp16_acoustic_modules=0,
                ar_coefficient_audit={'matrices': [{'block': i} for i in range(17)]}, runs=[row], warmups=[copy.deepcopy(row)],
                fft128_cin240={'modules': [{}]*6, 'calls': {str(i): 4 for i in range(6)}, 'baseline_library_files_unchanged': True})
    return data, {'exit_code': 0}, {'semantic': [{'tokens': 133, 'sha256': 'fixed'}]}


def test_distinct_pcm_roundoff_bound_accepts_only_known_same_model_result():
    data, guard, case = sample()
    m.validate(data, guard, case, 'fft128')
    for field, bad in [('max_pcm_delta', 2), ('snr_db', 91.9), ('same_shape', False)]:
        modified = copy.deepcopy(data)
        modified['runs'][0]['comparison'][field] = bad
        with pytest.raises(RuntimeError):
            m.validate(modified, guard, case, 'fft128')
    data['runs'][0]['pcm_sha256'] = m.PCM['baseline']
    with pytest.raises(RuntimeError, match='expected PCM'):
        m.validate(data, guard, case, 'fft128')


def test_warmup_tokens_and_cgroup_are_also_guarded():
    data, guard, case = sample()
    data['warmups'][0]['semantic'][0]['index'] = 132
    with pytest.raises(RuntimeError, match='semantic'):
        m.validate(data, guard, case, 'fft128')
    data, guard, case = sample()
    data['cgroup']['memory.swap.max'] = '1'
    with pytest.raises(RuntimeError, match='resource'):
        m.validate(data, guard, case, 'fft128')
