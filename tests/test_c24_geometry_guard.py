import copy
import pytest
from scripts.run_student_matrix import validate_c24_geometry


def test_c24_only_final_original_k11_branch_removed_and_ar17_bytes_preserved():
    parent = dict(decoder_branches=[[1,2],[1,2],[2],[1,2],[0,1,2]], ar_layers=17,
                  files={'ar.pt': {'sha256':'same', 'bytes':123}}, decoder_bottlenecks=[{'same':True}])
    candidate = copy.deepcopy(parent)
    candidate['decoder_branches'][-1] = [0,1]
    validate_c24_geometry(parent, candidate)
    for mutate in (lambda d:d['files']['ar.pt'].update(sha256='other'),
                   lambda d:d['decoder_branches'][0].pop(),
                   lambda d:d.update(ar_layers=16),
                   lambda d:d.update(decoder_bottlenecks=[])):
        wrong = copy.deepcopy(candidate)
        mutate(wrong)
        with pytest.raises(ValueError):
            validate_c24_geometry(parent, wrong)
